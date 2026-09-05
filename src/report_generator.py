import sqlite3
import json
import os
from datetime import datetime


class ReportGenerator:

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

        project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        self.report_dir = os.path.join(
            project_root,
            "reports"
        )

        os.makedirs(
            self.report_dir,
            exist_ok=True
        )

    def generate(self, session_id):

        connection = sqlite3.connect(
            self.database
        )

        connection.row_factory = sqlite3.Row

        cursor = connection.cursor()

        session = cursor.execute("""
            SELECT
                session_id,
                start_time,
                end_time,
                status
            FROM sessions
            WHERE session_id = ?
        """, (session_id,)).fetchone()

        if session is None:

            connection.close()

            print(
                f"Report could not be generated. "
                f"Session not found: {session_id}"
            )

            return None

        violations = cursor.execute("""
            SELECT
                id,
                type,
                timestamp,
                duration,
                confidence,
                severity,
                evidence_path
            FROM violations
            WHERE session_id = ?
            ORDER BY timestamp
        """, (session_id,)).fetchall()

        connection.close()

        start_time = datetime.strptime(
            session["start_time"],
            "%Y-%m-%d %H:%M:%S"
        )

        end_time = datetime.strptime(
            session["end_time"],
            "%Y-%m-%d %H:%M:%S"
        )

        total_duration = (
            end_time - start_time
        ).total_seconds()

        violation_summary = {}
        severity_summary = {}
        detailed_violations = []

        for violation in violations:

            violation_type = violation["type"]
            severity = violation["severity"]

            violation_summary[violation_type] = (
                violation_summary.get(
                    violation_type,
                    0
                ) + 1
            )

            severity_summary[severity] = (
                severity_summary.get(
                    severity,
                    0
                ) + 1
            )

            detailed_violations.append({
                "id": violation["id"],
                "type": violation["type"],
                "timestamp": violation["timestamp"],
                "duration": violation["duration"],
                "confidence": violation["confidence"],
                "severity": violation["severity"],
                "evidence_path": violation["evidence_path"]
            })

        if len(violations) == 0:

            final_result = "NO VIOLATIONS"

        elif any(
            violation["severity"] == "HIGH"
            for violation in violations
        ):

            final_result = "HIGH RISK"

        elif any(
            violation["severity"] == "MEDIUM"
            for violation in violations
        ):

            final_result = "MEDIUM RISK"

        else:

            final_result = "LOW RISK"

        report = {
            "session": {
                "session_id": session["session_id"],
                "start_time": session["start_time"],
                "end_time": session["end_time"],
                "status": session["status"],
                "total_duration_seconds": total_duration
            },
            "summary": {
                "total_violations": len(violations),
                "violation_types": violation_summary,
                "severity": severity_summary,
                "final_result": final_result
            },
            "violations": detailed_violations
        }

        report_path = os.path.join(
            self.report_dir,
            f"{session_id}.json"
        )

        with open(
            report_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                report,
                file,
                indent=4
            )

        print()
        print("=" * 60)
        print("EXAM REPORT GENERATED")
        print("=" * 60)
        print(f"Session ID       : {session_id}")
        print(f"Total Violations : {len(violations)}")
        print(f"Final Result     : {final_result}")
        print(f"Report           : {report_path}")
        print("=" * 60)
        print()

        return report_path