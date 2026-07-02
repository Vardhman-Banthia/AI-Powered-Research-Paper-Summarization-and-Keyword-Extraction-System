import sqlite3
import json
import os
from shared.constants import DATABASE_PATH

class PaperDatabase:
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._create_table()
        
    def _create_table(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    page_count INTEGER,
                    word_count INTEGER,
                    summary_short TEXT,
                    summary_medium TEXT,
                    summary_detailed TEXT,
                    keywords_json TEXT,
                    analytics_json TEXT
                )
            ''')
            conn.commit()
            
    def save_paper(self, filename: str, page_count: int, word_count: int, 
                   summary_short: str = None, summary_medium: str = None, 
                   summary_detailed: str = None, keywords: dict = None, 
                   analytics: dict = None) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO papers (filename, page_count, word_count, summary_short, summary_medium, summary_detailed, keywords_json, analytics_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                filename, 
                page_count, 
                word_count, 
                summary_short, 
                summary_medium, 
                summary_detailed, 
                json.dumps(keywords) if keywords else None, 
                json.dumps(analytics) if analytics else None
            ))
            conn.commit()
            return cursor.lastrowid
            
    def get_history(self, limit: int = 20) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM papers ORDER BY upload_date DESC LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]
            
    def get_paper(self, paper_id: int) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM papers WHERE id = ?', (paper_id,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                d['keywords'] = json.loads(d['keywords_json']) if d['keywords_json'] else {}
                d['analytics'] = json.loads(d['analytics_json']) if d['analytics_json'] else {}
                return d
            return None
            
    def delete_paper(self, paper_id: int) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM papers WHERE id = ?', (paper_id,))
            conn.commit()
            return cursor.rowcount > 0
            
    def update_paper(self, paper_id: int, **kwargs) -> bool:
        if not kwargs:
            return False
            
        fields = []
        values = []
        for k, v in kwargs.items():
            if k in ('keywords', 'analytics'):
                fields.append(f"{k}_json = ?")
                values.append(json.dumps(v))
            else:
                fields.append(f"{k} = ?")
                values.append(v)
                
        values.append(paper_id)
        
        query = f"UPDATE papers SET {', '.join(fields)} WHERE id = ?"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, values)
            conn.commit()
            return cursor.rowcount > 0
