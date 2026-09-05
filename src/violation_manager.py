import sqlite3
import os
from datetime import datetime


class ViolationManager:
    def __init__(self, database=None):

        if database is None:

            project_root = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )

            database = os.path.join(
                project_root,
                "database",
                "proctorvision.db"
            )

        self.database = database
        self.current_session_id = None

        os.makedirs(
            os.path.dirname(self.database),
            exist_ok=True
        )

        self.setup_database()


    def setup_database(self):

        connection = sqlite3.connect(
            self.database
        )

        cursor = connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                status TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                duration REAL,
                confidence REAL,
                severity TEXT NOT NULL,
                evidence_path TEXT
            )
        """)

        columns = [
            row[1]
            for row in cursor.execute(
                "PRAGMA table_info(violations)"
            ).fetchall()
        ]

        if "session_id" not in columns:
            cursor.execute("""
                ALTER TABLE violations
                ADD COLUMN session_id TEXT
            """)

        connection.commit()
        connection.close()


    def start_session(self):

        timestamp = datetime.now()
        session_id = timestamp.strftime(
            "EXAM_%Y%m%d_%H%M%S"
        )

        connection = sqlite3.connect(
            self.database
        )

        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO sessions
            (
                session_id,
                start_time,
                status
            )
            VALUES (?, ?, ?)
        """, (
            session_id,
            timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "RUNNING"
        ))

        connection.commit()
        connection.close()

        self.current_session_id = session_id

        print()
        print("=" * 60)
        print("EXAM SESSION STARTED")
        print("=" * 60)
        print(f"Session ID : {session_id}")
        print(f"Start Time : {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        print()

        return session_id


    def end_session(self):

        if self.current_session_id is None:
            return

        timestamp = datetime.now()

        connection = sqlite3.connect(
            self.database
        )

        cursor = connection.cursor()

        cursor.execute("""
            UPDATE sessions
            SET
                end_time = ?,
                status = ?
            WHERE session_id = ?
        """, (
            timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "COMPLETED",
            self.current_session_id
        ))

        connection.commit()
        connection.close()

        print()
        print("=" * 60)
        print("EXAM SESSION ENDED")
        print("=" * 60)
        print(f"Session ID : {self.current_session_id}")
        print(f"End Time   : {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        print()

        session_id = self.current_session_id
        self.current_session_id = None

        return session_id


    def report_violation(
        self,
        violation_type,
        duration,
        confidence,
        severity,
        evidence_path=None
    ):

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        connection = sqlite3.connect(
            self.database
        )

        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO violations
            (
                type,
                timestamp,
                duration,
                confidence,
                severity,
                evidence_path,
                session_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            violation_type,
            timestamp,
            duration,
            confidence,
            severity,
            evidence_path,
            self.current_session_id
        ))

        connection.commit()
        connection.close()

        print()
        print("=" * 60)
        print("VIOLATION DETECTED")
        print("=" * 60)
        print(f"Type       : {violation_type}")
        print(f"Duration   : {duration:.2f}s")
        print(f"Confidence : {confidence:.2%}")
        print(f"Severity   : {severity}")

        if evidence_path:
            print(f"Evidence   : {evidence_path}")

        if self.current_session_id:
            print(f"Session    : {self.current_session_id}")

        print("=" * 60)
        print()