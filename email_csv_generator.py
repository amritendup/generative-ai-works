import csv
import random
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()

output_file = "bfsi_email_data.csv"

# Compliance categories (non-compliant)
categories = [
    "Secrecy",
    "Market manipulation/Misconduct",
    "Market bribery",
    "Change in communication",
    "Complaints",
    "Employee ethics"
]

# Helper: assign category or mark as compliant
def assign_category():
    if random.random() < 0.3:  # 30% compliant, 70% non-compliant
        return "Compliant"
    else:
        return "; ".join(random.sample(categories, random.randint(1, 3)))

# Templates for non-compliant categories (no category leakage)
templates = {
    "Secrecy": [
        "Please keep the attached document restricted to internal teams only.",
        "Avoid circulating this investment report outside the department.",
        "Let’s ensure the attached data remains confidential within our group."
    ],
    "Market manipulation/Misconduct": [
        "Can you revise the figures slightly before sharing the final report?",
        "Let's align numbers based on tomorrow’s results before publishing.",
        "Please make the performance data appear more consistent."
    ],
    "Market bribery": [
        "The vendor mentioned an appreciation if we clear this quickly.",
        "We might get a favor if we approve their proposal early.",
        "They hinted at a reward for prioritizing their request."
    ],
    "Change in communication": [
        "Let’s continue this topic on a personal channel.",
        "We can discuss this over a call instead of email.",
        "Avoid extending this discussion over official mail."
    ],
    "Complaints": [
        "The client reported dissatisfaction with delayed service.",
        "Received escalation due to wrong debit entries on account.",
        "Customer mentioned poor response from the support team."
    ],
    "Employee ethics": [
        "Audit observed irregular entries in one associate’s activity.",
        "Concerns raised about a team member sharing internal data externally.",
        "We identified possible conflict of interest in client handling."
    ]
}

# Templates for compliant emails (normal business correspondence)
compliant_templates = [
    "Please find attached the monthly portfolio report for your review.",
    "The client meeting summary and minutes are attached for records.",
    "Kindly confirm the updated KYC document before next transaction.",
    "Here’s the project progress update as discussed in our call.",
    "Please review the attached reconciliation file and approve.",
    "Sharing internal memo for next week’s compliance meeting.",
    "Attached is the finalized interest rate sheet for the current quarter."
]

# Subject lines (neutral)
subjects = [
    "Internal update",
    "Follow-up on pending item",
    "Client review summary",
    "Monthly report attached",
    "Please verify the attached details",
    "Action required before submission",
    "Team coordination note",
    "Meeting notes and next steps"
]

# Generate BFSI-style email body
def generate_body(label):
    if label == "Compliant":
        body = random.choice(compliant_templates)
    else:
        parts = []
        for c in label.split("; "):
            if c in templates:
                parts.append(random.choice(templates[c]))
        # Add filler for realism
        parts += [
            fake.sentence(nb_words=10),
            fake.sentence(nb_words=12),
            "Please treat this message with appropriate discretion."
        ]
        body = " ".join(parts)
    return body

# Generate synthetic emails
records = []
for i in range(100):
    date = (datetime.now() - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d %H:%M:%S")
    from_email = fake.email()
    to_email = fake.email()
    label = assign_category()
    subject = random.choice(subjects)
    body = generate_body(label)
    records.append([date, from_email, to_email, subject, body, label])

# Write to CSV
with open(output_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Date", "From", "To", "Subject", "Body", "Categories"])
    writer.writerows(records)

print(f" Generated {len(records)} BFSI emails (compliant + non-compliant) in '{output_file}'")
