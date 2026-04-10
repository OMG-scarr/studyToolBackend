package llm

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type Client struct {
	baseURL    string
	chatModel  string
	embedModel string
	http       *http.Client
}

func NewClient(baseURL, chatModel, embedModel string) *Client {
	return &Client{
		baseURL:    baseURL,
		chatModel:  chatModel,
		embedModel: embedModel,
		http:       &http.Client{Timeout: 5 * time.Minute},
	}
}

// Embed returns the embedding vector for the given text.
func (c *Client) Embed(text string) ([]float64, error) {
	body := map[string]any{
		"model": c.embedModel,
		"input": text,
	}
	payload, _ := json.Marshal(body)

	resp, err := c.http.Post(c.baseURL+"/api/embed", "application/json", bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("ollama embed request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama embed error (%d): %s", resp.StatusCode, string(b))
	}

	var result struct {
		Embeddings [][]float64 `json:"embeddings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode embed response: %w", err)
	}
	if len(result.Embeddings) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}
	return result.Embeddings[0], nil
}

// EmbedBatch returns embeddings for multiple texts in one call.
func (c *Client) EmbedBatch(texts []string) ([][]float64, error) {
	body := map[string]any{
		"model": c.embedModel,
		"input": texts,
	}
	payload, _ := json.Marshal(body)

	resp, err := c.http.Post(c.baseURL+"/api/embed", "application/json", bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("ollama embed batch request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama embed batch error (%d): %s", resp.StatusCode, string(b))
	}

	var result struct {
		Embeddings [][]float64 `json:"embeddings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode embed batch response: %w", err)
	}
	return result.Embeddings, nil
}

// ChatMessage represents a message in the conversation.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Chat sends a question with context and returns the full answer (non-streaming).
func (c *Client) Chat(systemPrompt string, messages []ChatMessage) (string, error) {
	allMessages := make([]ChatMessage, 0, len(messages)+1)
	allMessages = append(allMessages, ChatMessage{Role: "system", Content: systemPrompt})
	allMessages = append(allMessages, messages...)

	body := map[string]any{
		"model":    c.chatModel,
		"messages": allMessages,
		"stream":   false,
		"options": map[string]any{
			"temperature": 0.3,
			"num_ctx":     2048,
		},
	}
	payload, _ := json.Marshal(body)

	resp, err := c.http.Post(c.baseURL+"/api/chat", "application/json", bytes.NewReader(payload))
	if err != nil {
		return "", fmt.Errorf("ollama chat request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("ollama chat error (%d): %s", resp.StatusCode, string(b))
	}

	var result struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("decode chat response: %w", err)
	}

	return stripThinkTags(result.Message.Content), nil
}

// StreamToken is a single token from the streaming response.
type StreamToken struct {
	Content string
	Done    bool
	Error   error
}

// ChatStream sends a question and streams tokens back via channel.
func (c *Client) ChatStream(systemPrompt string, messages []ChatMessage) <-chan StreamToken {
	ch := make(chan StreamToken, 64)

	go func() {
		defer close(ch)

		allMessages := make([]ChatMessage, 0, len(messages)+1)
		allMessages = append(allMessages, ChatMessage{Role: "system", Content: systemPrompt})
		allMessages = append(allMessages, messages...)

		body := map[string]any{
			"model":    c.chatModel,
			"messages": allMessages,
			"stream":   true,
			"options": map[string]any{
				"temperature": 0.3,
			},
		}
		payload, _ := json.Marshal(body)

		resp, err := c.http.Post(c.baseURL+"/api/chat", "application/json", bytes.NewReader(payload))
		if err != nil {
			ch <- StreamToken{Error: fmt.Errorf("ollama chat stream failed: %w", err)}
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			b, _ := io.ReadAll(resp.Body)
			ch <- StreamToken{Error: fmt.Errorf("ollama chat stream error (%d): %s", resp.StatusCode, string(b))}
			return
		}

		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

		for scanner.Scan() {
			line := scanner.Bytes()
			if len(line) == 0 {
				continue
			}

			var chunk struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
				Done bool `json:"done"`
			}
			if err := json.Unmarshal(line, &chunk); err != nil {
				continue
			}

			ch <- StreamToken{
				Content: chunk.Message.Content,
				Done:    chunk.Done,
			}

			if chunk.Done {
				return
			}
		}

		if err := scanner.Err(); err != nil {
			ch <- StreamToken{Error: err}
		}
	}()

	return ch
}

// Generate performs a one-shot generation (for content generation tasks).
func (c *Client) Generate(systemPrompt, userPrompt string) (string, error) {
	messages := []ChatMessage{
		{Role: "user", Content: userPrompt},
	}
	return c.Chat(systemPrompt, messages)
}

func stripThinkTags(s string) string {
	for {
		start := strings.Index(s, "<think>")
		if start == -1 {
			break
		}
		end := strings.Index(s, "</think>")
		if end == -1 {
			break
		}
		s = s[:start] + s[end+len("</think>"):]
	}
	return strings.TrimSpace(s)
}
