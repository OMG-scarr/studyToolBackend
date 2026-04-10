package ollama

import (
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
		http:       &http.Client{Timeout: 10 * time.Minute},
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

// Chat sends a question with context to the LLM and returns the answer.
func (c *Client) Chat(question, context string) (string, error) {
	systemPrompt := `You are a helpful study assistant chatbot. Answer the user's question based ONLY on the provided context from their uploaded documents.

Rules:
- Be conversational and friendly
- Give clear, well-structured answers
- If the context doesn't contain enough information to fully answer, say what you found and note what's missing
- Reference the source documents when relevant
- Do NOT make up information that isn't in the context

Context from uploaded documents:
` + context

	body := map[string]any{
		"model": c.chatModel,
		"messages": []map[string]string{
			{"role": "system", "content": systemPrompt},
			{"role": "user", "content": question},
		},
		"stream": false,
		"options": map[string]any{
			"temperature": 0.3,
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

	// Strip <think>...</think> tags from deepseek-r1 reasoning
	answer := result.Message.Content
	for {
		start := strings.Index(answer, "<think>")
		if start == -1 {
			break
		}
		end := strings.Index(answer, "</think>")
		if end == -1 {
			break
		}
		answer = answer[:start] + answer[end+len("</think>"):]
	}

	return strings.TrimSpace(answer), nil
}
