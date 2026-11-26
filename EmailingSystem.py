
from dotenv import load_dotenv, find_dotenv
import os
import smtplib
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

load_dotenv(find_dotenv())

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

BASE_DIR = Path(__file__).resolve().parent

def email_user(email, name, message):
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        raise ValueError("Email credentials are missing.")

    msg = MIMEMultipart()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = email
    msg["Subject"] = "AI Fitness Trainer - Performance Summary"

    body = f"""
    <html>
    <body>
        <h2 style="color:#7656d8;">Amazing Workout, {name}!</h2>
        <p>Here is your performance summary:</p>
        <p><b>{message}</b></p>
        <p>Keep exercising and stay healthy!</p>
        <p>AI Fitness Trainer</p>
    </body>
    </html>
    """
    msg.attach(MIMEText(body, "html"))

    # Attach an image only if one is available.
    for filename in ["fitness.jpg", "fitness1.jpg",
                     "fitness2.jpg", "fitness3.jpg"]:
        image_path = BASE_DIR / filename
        if image_path.exists():
            with open(image_path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f'attachment; filename="{filename}"'
                )
                msg.attach(part)
            break

    with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)

    return True
