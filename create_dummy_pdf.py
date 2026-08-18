from fpdf import FPDF

content = """
Company Leave Policy

All employees are entitled to 18 days of paid annual leave per calendar year.
Leave requests must be submitted at least 3 days in advance through the HR portal.
Unused leave can be carried forward to the next year, up to a maximum of 5 days.

Sick Leave Policy

Employees are entitled to 12 days of paid sick leave per year.
A medical certificate is required for sick leave exceeding 2 consecutive days.
Sick leave cannot be carried forward to the next year.

Work From Home Policy

Employees may work from home up to 2 days per week with manager approval.
Full-time remote work requires special approval from department head.
Employees must be available on company communication tools during work hours.

Reimbursement Policy

Travel expenses for official work purposes will be reimbursed within 30 days.
Employees must submit original receipts along with the reimbursement form.
Maximum reimbursement for local travel is capped at 500 rupees per day.
"""

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

for line in content.strip().split("\n"):
    pdf.multi_cell(0, 10, line)

pdf.output("data/sample.pdf")
print("Dummy PDF created at data/sample.pdf")