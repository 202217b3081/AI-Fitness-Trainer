import os
import smtplib
from dotenv import load_dotenv
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

load_dotenv()

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

def email_user(email, name, message):

    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        raise ValueError("Email credentials are missing")

    msg = MIMEMultipart()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = email
    msg["Subject"] = "Performance Summary"

    body = f"""
    <h2>Amazing Workout {name}!</h2>
    <p>Here is your Performance Summary:</p>
    <p>{message}</p>
    """

    msg.attach(MIMEText(body, "html"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)

    return True
