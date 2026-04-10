package chunker

import "strings"

// Chunk splits text into overlapping chunks for embedding.
func Chunk(text string, chunkSize, overlap int) []string {
	if chunkSize <= 0 {
		chunkSize = 1000
	}
	if overlap < 0 {
		overlap = 0
	}
	if overlap >= chunkSize {
		overlap = chunkSize / 5
	}

	text = strings.TrimSpace(text)
	if len(text) == 0 {
		return nil
	}

	// Try to split on paragraph boundaries first
	separators := []string{"\n\n", "\n", ". ", " "}
	return recursiveSplit(text, separators, chunkSize, overlap)
}

func recursiveSplit(text string, separators []string, chunkSize, overlap int) []string {
	if len(text) <= chunkSize {
		trimmed := strings.TrimSpace(text)
		if trimmed == "" {
			return nil
		}
		return []string{trimmed}
	}

	// Find the best separator
	sep := ""
	for _, s := range separators {
		if strings.Contains(text, s) {
			sep = s
			break
		}
	}

	if sep == "" {
		// No separator found, hard split
		return hardSplit(text, chunkSize, overlap)
	}

	parts := strings.Split(text, sep)
	var chunks []string
	var current strings.Builder

	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		candidate := current.String()
		if candidate != "" {
			candidate += sep + part
		} else {
			candidate = part
		}

		if len(candidate) > chunkSize && current.Len() > 0 {
			chunks = append(chunks, strings.TrimSpace(current.String()))

			// Overlap: keep the tail of the current chunk
			prev := current.String()
			current.Reset()
			if overlap > 0 && len(prev) > overlap {
				tail := prev[len(prev)-overlap:]
				// Find a clean break point in the tail
				if idx := strings.Index(tail, " "); idx != -1 {
					tail = tail[idx+1:]
				}
				current.WriteString(tail)
				current.WriteString(sep)
			}
			current.WriteString(part)
		} else {
			current.Reset()
			current.WriteString(candidate)
		}
	}

	if current.Len() > 0 {
		final := strings.TrimSpace(current.String())
		if final != "" {
			chunks = append(chunks, final)
		}
	}

	return chunks
}

func hardSplit(text string, chunkSize, overlap int) []string {
	var chunks []string
	step := chunkSize - overlap
	if step <= 0 {
		step = chunkSize
	}

	for i := 0; i < len(text); i += step {
		end := i + chunkSize
		if end > len(text) {
			end = len(text)
		}
		chunk := strings.TrimSpace(text[i:end])
		if chunk != "" {
			chunks = append(chunks, chunk)
		}
		if end == len(text) {
			break
		}
	}
	return chunks
}
