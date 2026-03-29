from __future__ import annotations

from datetime import date, timedelta
import random


def generate_employee_dataset(size: int = 120, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    departments = ["engineering", "finance", "sales", "hr", "it", "support"]
    first_names = ["Alex", "Taylor", "Jordan", "Sam", "Avery", "Riley", "Casey", "Morgan"]
    last_names = ["Ng", "Patel", "Diaz", "Singh", "Kim", "Brown", "Chen", "Smith"]
    start_anchor = date(2017, 1, 1)

    records: list[dict] = []
    for i in range(size):
        full_name = f"{rng.choice(first_names)} {rng.choice(last_names)}"
        records.append(
            {
                "employee_id": 1000 + i,
                "name": full_name,
                "department": rng.choice(departments),
                "salary": rng.randint(55_000, 180_000),
                "start_date": (start_anchor + timedelta(days=rng.randint(0, 3000))).isoformat(),
            }
        )

    return records