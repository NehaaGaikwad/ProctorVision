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

        connection.commit()
        connection.close()


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
                evidence_path
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            violation_type,
            timestamp,
            duration,
            confidence,
            severity,
            evidence_path
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

        print("=" * 60)
        print()