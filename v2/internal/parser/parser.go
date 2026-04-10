package parser

import (
	"fmt"
	"io"
	"path/filepath"
	"strings"

	"github.com/ledongthuc/pdf"
	"github.com/nguyenthenguyen/docx"
)

// Parse reads a file and returns its text content.
func Parse(filename string, r io.ReaderAt, size int64) (string, error) {
	ext := strings.ToLower(filepath.Ext(filename))
	switch ext {
	case ".pdf":
		return parsePDF(r, size)
	case ".docx":
		return parseDocx(r, size)
	case ".txt", ".md", ".csv":
		return parsePlainText(r, size)
	default:
		return "", fmt.Errorf("unsupported file type: %s", ext)
	}
}

func parsePDF(r io.ReaderAt, size int64) (string, error) {
	reader, err := pdf.NewReader(r, size)
	if err != nil {
		return "", fmt.Errorf("open PDF: %w", err)
	}

	var sb strings.Builder
	for i := 1; i <= reader.NumPage(); i++ {
		page := reader.Page(i)
		if page.V.IsNull() {
			continue
		}
		text, err := page.GetPlainText(nil)
		if err != nil {
			continue
		}
		sb.WriteString(text)
		sb.WriteString("\n\n")
	}

	return sb.String(), nil
}

func parseDocx(r io.ReaderAt, size int64) (string, error) {
	doc, err := docx.ReadDocxFromMemory(r, size)
	if err != nil {
		return "", fmt.Errorf("open DOCX: %w", err)
	}
	defer doc.Close()

	return doc.Editable().GetContent(), nil
}

func parsePlainText(r io.ReaderAt, size int64) (string, error) {
	buf := make([]byte, size)
	_, err := r.ReadAt(buf, 0)
	if err != nil && err != io.EOF {
		return "", fmt.Errorf("read text: %w", err)
	}
	return string(buf), nil
}
