package web

import (
	"html/template"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

type Renderer struct {
	mu    sync.RWMutex
	pages map[string]*template.Template
	dir   string
	dev   bool
}

var funcMap = template.FuncMap{
	"truncate": func(s string, n int) string {
		if len(s) <= n {
			return s
		}
		return s[:n] + "..."
	},
	"pct": func(f float64) int {
		return int(f * 100)
	},
	"list": func(items ...string) []string {
		return items
	},
}

func NewRenderer(templateDir string, dev bool) *Renderer {
	r := &Renderer{dir: templateDir, dev: dev, pages: make(map[string]*template.Template)}
	r.load()
	return r
}

func (r *Renderer) load() {
	layoutFile := filepath.Join(r.dir, "layout.html")

	// Collect all partial files
	partialFiles, _ := filepath.Glob(filepath.Join(r.dir, "partials", "*.html"))

	// Base files = layout + all partials
	baseFiles := append([]string{layoutFile}, partialFiles...)

	// Each page template gets its own clone with layout + partials
	pageFiles, _ := filepath.Glob(filepath.Join(r.dir, "*.html"))

	pages := make(map[string]*template.Template)
	for _, pageFile := range pageFiles {
		name := strings.TrimSuffix(filepath.Base(pageFile), ".html")
		if name == "layout" {
			continue
		}

		files := append(append([]string{}, baseFiles...), pageFile)
		t, err := template.New("").Funcs(funcMap).ParseFiles(files...)
		if err != nil {
			log.Printf("Failed to parse template %s: %v", name, err)
			continue
		}
		pages[name] = t
	}

	r.mu.Lock()
	r.pages = pages
	r.mu.Unlock()
}

// Render executes a page template by name (e.g. "home", "notebook").
func (r *Renderer) Render(w io.Writer, name string, data any) error {
	if r.dev {
		r.load()
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	t, ok := r.pages[name]
	if !ok {
		log.Printf("template not found: %s", name)
		return os.ErrNotExist
	}

	// Execute the "layout" template which pulls in the page's content/title blocks
	if err := t.ExecuteTemplate(w, "layout", data); err != nil {
		log.Printf("template render error (%s): %v", name, err)
		return err
	}
	return nil
}

// RenderPartial executes a named partial/define block.
func (r *Renderer) RenderPartial(w io.Writer, pageName, blockName string, data any) error {
	if r.dev {
		r.load()
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	// Use any page template since partials are shared
	for _, t := range r.pages {
		if err := t.ExecuteTemplate(w, blockName, data); err == nil {
			return nil
		}
	}

	// Try the specific page
	if t, ok := r.pages[pageName]; ok {
		return t.ExecuteTemplate(w, blockName, data)
	}

	return os.ErrNotExist
}
