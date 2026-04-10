package database

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	_ "modernc.org/sqlite"

	"studytool/v2/internal/models"
)

type DB struct {
	conn *sql.DB
}

func New(dataDir string) (*DB, error) {
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		return nil, fmt.Errorf("create data dir: %w", err)
	}

	dbPath := filepath.Join(dataDir, "studytool.db")
	conn, err := sql.Open("sqlite", dbPath+"?_journal_mode=WAL&_foreign_keys=on")
	if err != nil {
		return nil, fmt.Errorf("open database: %w", err)
	}

	db := &DB{conn: conn}
	if err := db.migrate(); err != nil {
		return nil, fmt.Errorf("migrate: %w", err)
	}

	return db, nil
}

func (db *DB) Close() error {
	return db.conn.Close()
}

func (db *DB) migrate() error {
	schema := `
	CREATE TABLE IF NOT EXISTS notebooks (
		id          TEXT PRIMARY KEY,
		title       TEXT NOT NULL,
		description TEXT NOT NULL DEFAULT '',
		emoji       TEXT NOT NULL DEFAULT '📓',
		created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
		updated_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
	);

	CREATE TABLE IF NOT EXISTS sources (
		id          TEXT PRIMARY KEY,
		notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
		filename    TEXT NOT NULL,
		file_type   TEXT NOT NULL,
		file_size   INTEGER NOT NULL DEFAULT 0,
		chunk_count INTEGER NOT NULL DEFAULT 0,
		summary     TEXT NOT NULL DEFAULT '',
		created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
	);

	CREATE TABLE IF NOT EXISTS chat_messages (
		id          TEXT PRIMARY KEY,
		notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
		role        TEXT NOT NULL,
		content     TEXT NOT NULL,
		sources_json TEXT NOT NULL DEFAULT '[]',
		created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
	);

	CREATE TABLE IF NOT EXISTS generated_content (
		id          TEXT PRIMARY KEY,
		notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
		type        TEXT NOT NULL,
		title       TEXT NOT NULL,
		content     TEXT NOT NULL,
		created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
	);

	CREATE INDEX IF NOT EXISTS idx_sources_notebook ON sources(notebook_id);
	CREATE INDEX IF NOT EXISTS idx_chat_notebook ON chat_messages(notebook_id);
	CREATE INDEX IF NOT EXISTS idx_generated_notebook ON generated_content(notebook_id);
	`
	_, err := db.conn.Exec(schema)
	return err
}

// --- Notebooks ---

func (db *DB) CreateNotebook(nb *models.Notebook) error {
	_, err := db.conn.Exec(
		`INSERT INTO notebooks (id, title, description, emoji, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)`,
		nb.ID, nb.Title, nb.Description, nb.Emoji, nb.CreatedAt, nb.UpdatedAt,
	)
	return err
}

func (db *DB) GetNotebook(id string) (*models.Notebook, error) {
	nb := &models.Notebook{}
	err := db.conn.QueryRow(`
		SELECT n.id, n.title, n.description, n.emoji, n.created_at, n.updated_at,
			   COALESCE((SELECT COUNT(*) FROM sources WHERE notebook_id = n.id), 0),
			   COALESCE((SELECT SUM(chunk_count) FROM sources WHERE notebook_id = n.id), 0)
		FROM notebooks n WHERE n.id = ?`, id,
	).Scan(&nb.ID, &nb.Title, &nb.Description, &nb.Emoji, &nb.CreatedAt, &nb.UpdatedAt,
		&nb.SourceCount, &nb.ChunkCount)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	return nb, err
}

func (db *DB) ListNotebooks() ([]models.Notebook, error) {
	rows, err := db.conn.Query(`
		SELECT n.id, n.title, n.description, n.emoji, n.created_at, n.updated_at,
			   COALESCE((SELECT COUNT(*) FROM sources WHERE notebook_id = n.id), 0),
			   COALESCE((SELECT SUM(chunk_count) FROM sources WHERE notebook_id = n.id), 0)
		FROM notebooks n ORDER BY n.updated_at DESC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var notebooks []models.Notebook
	for rows.Next() {
		var nb models.Notebook
		if err := rows.Scan(&nb.ID, &nb.Title, &nb.Description, &nb.Emoji, &nb.CreatedAt, &nb.UpdatedAt,
			&nb.SourceCount, &nb.ChunkCount); err != nil {
			return nil, err
		}
		notebooks = append(notebooks, nb)
	}
	return notebooks, rows.Err()
}

func (db *DB) UpdateNotebook(nb *models.Notebook) error {
	nb.UpdatedAt = time.Now()
	_, err := db.conn.Exec(
		`UPDATE notebooks SET title = ?, description = ?, emoji = ?, updated_at = ? WHERE id = ?`,
		nb.Title, nb.Description, nb.Emoji, nb.UpdatedAt, nb.ID,
	)
	return err
}

func (db *DB) DeleteNotebook(id string) error {
	_, err := db.conn.Exec(`DELETE FROM notebooks WHERE id = ?`, id)
	return err
}

func (db *DB) TouchNotebook(id string) error {
	_, err := db.conn.Exec(`UPDATE notebooks SET updated_at = ? WHERE id = ?`, time.Now(), id)
	return err
}

// --- Sources ---

func (db *DB) CreateSource(s *models.Source) error {
	_, err := db.conn.Exec(
		`INSERT INTO sources (id, notebook_id, filename, file_type, file_size, chunk_count, summary, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		s.ID, s.NotebookID, s.Filename, s.FileType, s.FileSize, s.ChunkCount, s.Summary, s.CreatedAt,
	)
	return err
}

func (db *DB) ListSources(notebookID string) ([]models.Source, error) {
	rows, err := db.conn.Query(
		`SELECT id, notebook_id, filename, file_type, file_size, chunk_count, summary, created_at FROM sources WHERE notebook_id = ? ORDER BY created_at DESC`,
		notebookID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var sources []models.Source
	for rows.Next() {
		var s models.Source
		if err := rows.Scan(&s.ID, &s.NotebookID, &s.Filename, &s.FileType, &s.FileSize, &s.ChunkCount, &s.Summary, &s.CreatedAt); err != nil {
			return nil, err
		}
		sources = append(sources, s)
	}
	return sources, rows.Err()
}

func (db *DB) GetSource(id string) (*models.Source, error) {
	s := &models.Source{}
	err := db.conn.QueryRow(
		`SELECT id, notebook_id, filename, file_type, file_size, chunk_count, summary, created_at FROM sources WHERE id = ?`, id,
	).Scan(&s.ID, &s.NotebookID, &s.Filename, &s.FileType, &s.FileSize, &s.ChunkCount, &s.Summary, &s.CreatedAt)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	return s, err
}

func (db *DB) DeleteSource(id string) error {
	_, err := db.conn.Exec(`DELETE FROM sources WHERE id = ?`, id)
	return err
}

func (db *DB) GetSourceIDsByNotebook(notebookID string) ([]string, error) {
	rows, err := db.conn.Query(`SELECT id FROM sources WHERE notebook_id = ?`, notebookID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	return ids, rows.Err()
}

// --- Chat Messages ---

func (db *DB) SaveChatMessage(msg *models.ChatMessage) error {
	sourcesJSON, _ := json.Marshal(msg.Sources)
	_, err := db.conn.Exec(
		`INSERT INTO chat_messages (id, notebook_id, role, content, sources_json, created_at) VALUES (?, ?, ?, ?, ?, ?)`,
		msg.ID, msg.NotebookID, msg.Role, msg.Content, string(sourcesJSON), msg.CreatedAt,
	)
	return err
}

func (db *DB) GetChatHistory(notebookID string, limit int) ([]models.ChatMessage, error) {
	if limit <= 0 {
		limit = 50
	}
	rows, err := db.conn.Query(
		`SELECT id, notebook_id, role, content, sources_json, created_at FROM chat_messages WHERE notebook_id = ? ORDER BY created_at ASC LIMIT ?`,
		notebookID, limit,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var messages []models.ChatMessage
	for rows.Next() {
		var msg models.ChatMessage
		var sourcesJSON string
		if err := rows.Scan(&msg.ID, &msg.NotebookID, &msg.Role, &msg.Content, &sourcesJSON, &msg.CreatedAt); err != nil {
			return nil, err
		}
		json.Unmarshal([]byte(sourcesJSON), &msg.Sources)
		messages = append(messages, msg)
	}
	return messages, rows.Err()
}

func (db *DB) ClearChatHistory(notebookID string) error {
	_, err := db.conn.Exec(`DELETE FROM chat_messages WHERE notebook_id = ?`, notebookID)
	return err
}

// --- Generated Content ---

func (db *DB) SaveGenerated(gc *models.GeneratedContent) error {
	_, err := db.conn.Exec(
		`INSERT INTO generated_content (id, notebook_id, type, title, content, created_at) VALUES (?, ?, ?, ?, ?, ?)`,
		gc.ID, gc.NotebookID, gc.Type, gc.Title, gc.Content, gc.CreatedAt,
	)
	return err
}

func (db *DB) ListGenerated(notebookID string) ([]models.GeneratedContent, error) {
	rows, err := db.conn.Query(
		`SELECT id, notebook_id, type, title, content, created_at FROM generated_content WHERE notebook_id = ? ORDER BY created_at DESC`,
		notebookID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var items []models.GeneratedContent
	for rows.Next() {
		var gc models.GeneratedContent
		if err := rows.Scan(&gc.ID, &gc.NotebookID, &gc.Type, &gc.Title, &gc.Content, &gc.CreatedAt); err != nil {
			return nil, err
		}
		items = append(items, gc)
	}
	return items, rows.Err()
}

func (db *DB) DeleteGenerated(id string) error {
	_, err := db.conn.Exec(`DELETE FROM generated_content WHERE id = ?`, id)
	return err
}
