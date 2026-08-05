def transform_row(row):
    
    # Cleans a single row of data and returns the cleaned version.


    cleaned_row = {}

    for key, value in row.items():

        # Handle missing values
        if value == "":
            cleaned_row[key] = None

        # Clean text values
        elif isinstance(value, str):
            cleaned_row[key] = value.strip()

        else:
            cleaned_row[key] = value

    # Standardize selected fields

    if cleaned_row.get("department"):
        cleaned_row["department"] = cleaned_row["department"].title()

    if cleaned_row.get("job_title"):
        cleaned_row["job_title"] = cleaned_row["job_title"].title()

    if cleaned_row.get("state"):
        cleaned_row["state"] = cleaned_row["state"].upper()

    if cleaned_row.get("email"):
        cleaned_row["email"] = cleaned_row["email"].lower()

    return cleaned_row
        