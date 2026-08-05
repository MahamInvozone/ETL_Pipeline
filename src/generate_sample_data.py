# ===============================================================================
# Script Name: generate_sample_data.py
# Description: Generates a 1,000-row x 50-column synthetic dataset simulating
#              realistic, dirty business and employee records for ETL pipeline practice.
# Dependencies: pandas, numpy, python standard libraries
# Output: data/sample_data.csv
# ===============================================================================

# COLUMN DOCUMENTATION TABLE
# ----------------------------------------------------------------------------------------------------------------------------------
# Column Name           | Data Type | Purpose                                | Intentional Dirty-Data Pattern
# ----------------------------------------------------------------------------------------------------------------------------------
# record_id             | Integer   | Unique primary key identifier          | Clean (Sequential primary key, 1 to 1000)
# employee_id           | String    | Standardized employee code             | Standardized string format (EMP-XXXX)
# first_name            | String    | Employee given name                    | Includes leading/trailing whitespace, mixed casing
# last_name             | String    | Employee family name                   | Includes leading/trailing whitespace
# full_name             | String    | Combined first and last name           | Inconsistent whitespace, missing values (~3%)
# gender                | String    | Gender identity                        | Inconsistent capitalization ("Male", "female", "M")
# age                   | Integer   | Age in years                           | Missing values (~4%), occasional negative/outlier values
# date_of_birth         | Date/Str  | Date of birth (YYYY-MM-DD)             | Occasional missing dates (~3%)
# email                 | String    | Work email address                     | Mixed casing ("USER@DOMAIN.COM"), trailing spaces
# phone_number          | String    | Contact phone number                   | Mixed formats (e.g., "555-123-4567", "(555) 123-4567")
# address               | String    | Street address                         | Includes trailing spaces, blank strings
# city                  | String    | Residence city                         | Inconsistent casing ("New york", "NEW YORK")
# state                 | String    | State abbreviation                     | Uppercase/Lowercase mix ("ca", "CA", "Ca")
# country               | String    | Country of employment                  | Inconsistent naming ("USA", "United States", "US")
# postal_code           | String    | ZIP / Postal code                      | Missing leading zeros, missing values
# department            | Categorical| Company department                    | Mixed casing ("HR", "hr", "Human Resources")
# job_title             | String    | Official job title                     | Leading/trailing whitespace
# manager_name          | String    | Direct supervisor name                 | Missing values (~5%) for top-level roles
# employment_status     | Categorical| Current employment status              | Inconsistent terms ("Active", "active", "FULL_TIME")
# work_location         | Categorical| Work setting (Onsite, Remote, Hybrid)  | Mixed casing ("Remote", "REMOTE", "remote")
# hire_date             | Date/Str  | Date joined company                    | Standard ISO date format
# last_promotion_date   | Date/Str  | Date of most recent promotion          | Missing values for non-promoted staff (~20%)
# years_experience      | Float     | Total professional experience          | Missing values (~3%)
# education_level       | Categorical| Highest degree attained                | Inconsistent labels ("Bachelors", "BS", "bachelor")
# salary                | Float     | Base annual compensation               | Missing values (~3%), non-standard rounding
# annual_bonus          | Float     | Yearly bonus target amount             | Missing values (~5%)
# tax_rate              | Float     | Applicable tax rate fraction           | Fixed decimal precision
# performance_score     | Integer   | Annual performance score (1-5)         | Missing values (~4%)
# attendance_percentage | Float     | Annual attendance rate (0-100%)        | Occasional out-of-bounds values (>100%)
# overtime_hours        | Float     | Overtime hours logged                  | Zero-inflated numeric values
# project_count         | Integer   | Number of assigned projects            | Standard integer distribution
# training_hours        | Integer   | Professional development hours logged   | Standard integer distribution
# certification_count  | Integer   | Total professional certifications held| Standard integer distribution
# vacation_days         | Integer   | Annual vacation days allocated         | Standard integer distribution
# sick_days             | Integer   | Annual sick leave days used            | Standard integer distribution
# employee_rating       | Categorical| Qualitative assessment level          | Inconsistent casing ("EXCELLENT", "Good", "low")
# bank_name             | String    | Direct deposit institution name        | Mixed casing and abbreviations
# account_type          | Categorical| Bank account type (Checking/Savings)   | Mixed casing ("Checking", "checking")
# emergency_contact     | String    | Primary emergency contact name         | Missing values (~5%)
# emergency_phone       | String    | Primary emergency contact phone        | Inconsistent phone formats
# marital_status        | Categorical| Civil status                           | Mixed casing ("Single", "married", "DIVORCED")
# nationality           | String    | Country of citizenship                 | Standard categorical strings
# shift                 | Categorical| Work shift schedule                    | Inconsistent naming ("Day", "night", "SHIFT 1")
# contract_type         | Categorical| Employment agreement classification    | Mixed casing ("Full-Time", "contract", "FT")
# office_floor          | Integer   | Floor assignment                       | Missing values for remote staff (~15%)
# workstation_id        | String    | Desk / Cubicle identifier              | Missing values, leading/trailing spaces
# laptop_assigned       | Boolean   | Whether company laptop is assigned     | Boolean with missing values (~3%)
# insurance_provider    | Categorical| Benefits insurance carrier             | Inconsistent company naming
# remarks               | String    | Free-text HR notes                     | Contains blank strings and "N/A" text entries
# last_updated          | Date/Str  | System audit timestamp                 | ISO timestamp format
# ----------------------------------------------------------------------------------------------------------------------------------

from datetime import datetime, timedelta
from pathlib import Path
import random

import numpy as np
import pandas as pd

# =============================================================================
# 1. SETUP & REPRODUCIBILITY
# =============================================================================
# Set random seeds for NumPy and Python's built-in random module for repeatability.
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

NUM_ROWS = 1000

# =============================================================================
# 2. RAW SEED DATA POOLS
# =============================================================================
FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
    "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph",
    "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy",
    "Daniel", "Lisa", "Matthew", "Margaret", "Anthony", "Betty", "Mark", "Sandra"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson",
    "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee",
    "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez",
    "Lewis", "Robinson"
]

CITIES_STATES = [
    ("New York", "NY"), ("Los Angeles", "CA"), ("Chicago", "IL"),
    ("Houston", "TX"), ("Phoenix", "AZ"), ("Philadelphia", "PA"),
    ("San Antonio", "TX"), ("San Diego", "CA"), ("Dallas", "TX"),
    ("San Jose", "CA"), ("Austin", "TX"), ("Seattle", "WA"),
    ("Denver", "CO"), ("Boston", "MA"), ("Atlanta", "GA")
]

DEPARTMENTS = [
    "Engineering", "Human Resources", "Finance", "Marketing", "Sales",
    "Customer Support", "Product", "Legal", "Operations", "IT"
]

JOB_TITLES = [
    "Software Engineer", "Data Analyst", "HR Specialist", "Accountant",
    "Marketing Specialist", "Sales Executive", "Support Agent", "Product Manager",
    "Legal Counsel", "Operations Manager", "Systems Administrator", "Senior Developer"
]

BANKS = ["Chase", "Bank of America", "Wells Fargo", "Citibank", "Capital One", "TD Bank"]
INSURANCE_PROVIDERS = ["Blue Cross Blue Shield", "Aetna", "UnitedHealth", "Cigna", "Kaiser Permanente"]

# =============================================================================
# 3. HELPER FUNCTIONS FOR INTENTIONAL DATA DIRTINESS
# =============================================================================

def apply_whitespace(value: str, p: float = 0.15) -> str:
    # Randomly adds leading or trailing whitespace to text values.
    if not isinstance(value, str):
        return value
    if random.random() < p:
        prefix = " " * random.randint(1, 3) if random.random() > 0.5 else ""
        suffix = " " * random.randint(1, 3) if random.random() > 0.5 else ""
        return f"{prefix}{value}{suffix}"
    return value


def apply_casing_noise(value: str, p: float = 0.20) -> str:
    # Randomly mutates string casing (UPPERCASE, lowercase, TitleCase).
    if not isinstance(value, str):
        return value
    r = random.random()
    if r < p * 0.4:
        return value.lower()
    elif r < p * 0.8:
        return value.upper()
    elif r < p:
        return value.swapcase()
    return value


def add_missing(val, missing_val=np.nan, p: float = 0.04, string_blank: bool = False):
    # Inject missing values (NaN or empty strings) with probability p.
    if random.random() < p:
        return "" if string_blank and random.random() < 0.5 else missing_val
    return val


def random_date(start_year=2015, end_year=2025) -> str:
    # Generates a random date string formatted as YYYY-MM-DD.
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    random_days = random.randint(0, delta.days)
    return (start + timedelta(days=random_days)).strftime("%Y-%m-%d")


# =============================================================================
# 4. DATA GENERATION LOGIC
# =============================================================================
print("Generating synthetic records...")

data = []
for i in range(1, NUM_ROWS + 1):
    rec_id = i
    emp_id = f"EMP-{random.randint(10000, 99999)}"

    # Names
    fn = random.choice(FIRST_NAMES)
    ln = random.choice(LAST_NAMES)

    fn_dirty = apply_whitespace(fn, p=0.10)
    ln_dirty = apply_whitespace(ln, p=0.10)

    if random.random() < 0.03:
        full_n = np.nan
    else:
        full_n = apply_casing_noise(f"{fn} {ln}", p=0.15)

    # Gender with category variants
    gender_raw = random.choice(["Male", "Female", "Non-Binary", "Prefer Not to Say"])
    gender = apply_casing_noise(gender_raw, p=0.25)

    # Age & DOB
    age_val = int(np.random.normal(38, 10))
    age_val = max(18, min(68, age_val))
    age = add_missing(age_val, p=0.04)

    dob_year = 2026 - int(age_val) if not pd.isna(age_val) else 1988
    dob = random_date(start_year=dob_year - 1, end_year=dob_year)
    dob = add_missing(dob, p=0.03)

    # Email
    domain = random.choice(["company.com", "enterprise.org", "corp.net"])
    email_clean = f"{fn.lower()}.{ln.lower()}@{domain}"
    email = apply_casing_noise(email_clean, p=0.20)
    if random.random() < 0.10:
        email = f" {email} "

    # Contact Details
    phone = (
        f"{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
        if random.random() > 0.3
        else f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}"
    )
    city, state_abbr = random.choice(CITIES_STATES)

    city_dirty = apply_casing_noise(city, p=0.15)
    state_dirty = state_abbr.lower() if random.random() < 0.3 else state_abbr
    country_dirty = random.choice(["USA", "United States", "US", "usa", "U.S.A."])
    street_num = random.randint(100, 9999)
    street_name = random.choice(["Main St", "Oak Ave", "Maple Rd", "Washington Blvd", "Park Ave"])
    address = apply_whitespace(f"{street_num} {street_name}", p=0.10)
    postal_code = add_missing(str(random.randint(10000, 99999)), p=0.03)

    # Department & Position
    dept = random.choice(DEPARTMENTS)
    if dept == "Human Resources" and random.random() < 0.4:
        dept = random.choice(["HR", "hr", "human resources"])
    else:
        dept = apply_casing_noise(dept, p=0.15)

    job_title = apply_whitespace(random.choice(JOB_TITLES), p=0.12)
    manager = add_missing(f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}", p=0.05)

    # Status & Logistics
    emp_status = random.choice(["Active", "Active", "Active", "On Leave", "Terminated", "ACTIVE", "active"])
    work_loc = random.choice(["Onsite", "Remote", "Hybrid", "remote", "REMOTE", "hybrid"])

    hire_d = random_date(start_year=2012, end_year=2024)
    promo_d = random_date(start_year=2020, end_year=2025) if random.random() > 0.4 else np.nan

    exp = add_missing(round(max(0.5, np.random.normal(8, 5)), 1), p=0.03)
    edu = random.choice(["Bachelors", "Bachelor's", "BS", "bachelor", "Masters", "Master's", "MS", "PhD", "High School"])

    # Financial Metrics
    salary_base = float(np.random.normal(75000, 20000))
    salary_base = round(max(35000.0, salary_base), 2)
    salary = add_missing(salary_base, p=0.04)

    bonus_val = round(salary_base * random.uniform(0.05, 0.20), 2)
    annual_bonus = add_missing(bonus_val, p=0.05)
    tax_rate = round(random.choice([0.15, 0.22, 0.24, 0.32]), 2)

    # Performance & Activity
    perf_score = add_missing(random.choice([1, 2, 3, 4, 5]), p=0.04)
    attend_pct = round(min(100.0, max(60.0, np.random.normal(95, 5))), 1)
    if random.random() < 0.02:
        attend_pct = 105.0

    ot_hours = round(float(np.random.exponential(10)), 1) if random.random() > 0.3 else 0.0
    proj_count = random.randint(1, 10)
    train_hrs = random.randint(0, 80)
    cert_count = random.randint(0, 5)

    vacation = random.randint(5, 30)
    sick_d = random.randint(0, 12)
    rating = random.choice(["EXCELLENT", "Excellent", "Good", "good", "Needs Improvement", "satisfactory"])

    # Banking & Admin
    bank = apply_casing_noise(random.choice(BANKS), p=0.15)
    acc_type = random.choice(["Checking", "Savings", "checking", "savings", "CHECKING"])
    emerg_contact = add_missing(f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}", p=0.05)
    emerg_phone = add_missing(f"{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}", p=0.05)

    marital = random.choice(["Single", "Married", "Divorced", "single", "MARRIED"])
    nationality = random.choice(["American", "Canadian", "British", "Indian", "Mexican", "German"])

    shift = random.choice(["Day", "Night", "Shift 1", "Shift 2", "day", "NIGHT"])
    contract = random.choice(["Full-Time", "Part-Time", "Contract", "FT", "PT", "contract"])

    # Workstation Setup
    if str(work_loc).lower() == "remote":
        floor = np.nan
        ws_id = np.nan
    else:
        floor = random.randint(1, 15)
        ws_id = apply_whitespace(f"WS-{floor:02d}-{random.randint(10, 99)}", p=0.15)

    laptop = add_missing(random.choice([True, False]), p=0.03)
    insurance = random.choice(INSURANCE_PROVIDERS)

    # Remarks
    remark_options = [
        "High potential", "Pending annual review", "N/A", "",
        "Transferred from regional branch", "Completed leadership onboarding"
    ]
    remarks = add_missing(random.choice(remark_options), p=0.10, string_blank=True)
    last_upd = (datetime.now() - timedelta(days=random.randint(0, 60))).strftime("%Y-%m-%d %H:%M:%S")

    # Assemble row dictionary matching 50 columns
    row = {
        "record_id": rec_id,
        "employee_id": emp_id,
        "first_name": fn_dirty,
        "last_name": ln_dirty,
        "full_name": full_n,
        "gender": gender,
        "age": age,
        "date_of_birth": dob,
        "email": email,
        "phone_number": phone,
        "address": address,
        "city": city_dirty,
        "state": state_dirty,
        "country": country_dirty,
        "postal_code": postal_code,
        "department": dept,
        "job_title": job_title,
        "manager_name": manager,
        "employment_status": emp_status,
        "work_location": work_loc,
        "hire_date": hire_d,
        "last_promotion_date": promo_d,
        "years_experience": exp,
        "education_level": edu,
        "salary": salary,
        "annual_bonus": annual_bonus,
        "tax_rate": tax_rate,
        "performance_score": perf_score,
        "attendance_percentage": attend_pct,
        "overtime_hours": ot_hours,
        "project_count": proj_count,
        "training_hours": train_hrs,
        "certification_count": cert_count,
        "vacation_days": vacation,
        "sick_days": sick_d,
        "employee_rating": rating,
        "bank_name": bank,
        "account_type": acc_type,
        "emergency_contact": emerg_contact,
        "emergency_phone": emerg_phone,
        "marital_status": marital,
        "nationality": nationality,
        "shift": shift,
        "contract_type": contract,
        "office_floor": floor,
        "workstation_id": ws_id,
        "laptop_assigned": laptop,
        "insurance_provider": insurance,
        "remarks": remarks,
        "last_updated": last_upd,
    }
    data.append(row)

# Convert list of dicts to DataFrame
df = pd.DataFrame(data)

# Inject controlled duplicate records (copy 10 rows with new primary keys)
duplicate_indices = random.sample(range(len(df)), 10)
duplicate_rows = df.iloc[duplicate_indices].copy()
duplicate_rows["record_id"] = range(NUM_ROWS + 1, NUM_ROWS + 11)
df = pd.concat([df, duplicate_rows], ignore_index=True)


# =============================================================================
# 5. EXPORT AND VERIFICATION
# =============================================================================
# Automatically create the destination folder if it does not exist
output_dir = Path("data")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "sample_data.csv"
df.to_csv(output_path, index=False)

# Display summary output
print("\n" + "=" * 60)
print("DATASET GENERATION COMPLETE")
print("=" * 60)
print(f"Number of Rows:    {df.shape[0]}")
print(f"Number of Columns: {df.shape[1]}")
print(f"File Output Path:  {output_path.resolve()}")
print("-" * 60)
print("PREVIEW OF FIRST 5 ROWS:")
print("-" * 60)
print(df.head())