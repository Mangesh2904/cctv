import cv2
import torch
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
import numpy as np
import time

def get_cooldown_time():
    while True:
        try:
            cooldown_time = int(input("Enter cooldown time between emails (in seconds): "))
            if cooldown_time <= 0:
                print("Please enter a positive number.")
                continue
            return cooldown_time
        except ValueError:
            print("Please enter a valid number.")

def send_email_with_attachment(to_address, subject, body, attachment_path):
    from_address = ' '  # Replace with your Gmail
    email_password = ' '  # Replace with your Gmail App password

    # Create the email message
    msg = MIMEMultipart()
    msg['From'] = from_address
    msg['To'] = to_address
    msg['Subject'] = subject

    # Attach the body text
    msg.attach(MIMEText(body, 'plain'))

    # Attach the image file
    if os.path.exists(attachment_path):
        with open(attachment_path, 'rb') as attachment:
            mime_base = MIMEBase('application', 'octet-stream')
            mime_base.set_payload(attachment.read())
            encoders.encode_base64(mime_base)
            mime_base.add_header('Content-Disposition', f'attachment; filename= {os.path.basename(attachment_path)}')
            msg.attach(mime_base)
    else:
        print("Attachment not found!")

    # Send the email via Gmail's SMTP server
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_address, email_password)
        server.sendmail(from_address, to_address, msg.as_string())
        server.quit()
        print("Email sent successfully!")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

def main():
    print("Starting Human Detection System...")
    email_cooldown = get_cooldown_time()

    # Load YOLOv5 model
    print("Loading YOLOv5 model...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)

    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    email_sent = False  # Flag to avoid sending multiple emails
    last_email_time = 0  # Timestamp of last email sent

    print(f"\nSystem Ready!")
    print(f"Cooldown period between emails: {email_cooldown} seconds")
    print("Press 'q' to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Perform detection without rendering
        results = model(frame)

        # Get detection results and filter only humans (class 0 is person)
        detections = results.xyxy[0]
        human_detected = any(int(detection[5]) == 0 for detection in detections)

        # Create a copy of the original frame for display
        display_frame = frame.copy()

        current_time = time.time()

        if human_detected:
            status_text = "Human Detected!"
            color = (0, 255, 0)

            # Check if we should send an email
            if not email_sent and (current_time - last_email_time > email_cooldown):
                # Save the original frame (without boxes or text)
                image_path = 'detected_frame.jpg'
                cv2.imwrite(image_path, frame)

                # Send the email
                to_address = ' '  # Replace with recipient's email
                subject = "Alert: Human Detected"
                body = "A human has been detected in the monitored area."

                if send_email_with_attachment(to_address, subject, body, image_path):
                    email_sent = True
                    last_email_time = current_time
        else:
            status_text = "No Human Detected"
            color = (0, 0, 255)

        # Add status text to display frame
        cv2.putText(display_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Reset the email_sent flag after cooldown period
        if email_sent and (current_time - last_email_time > email_cooldown):
            email_sent = False

        # Show the webcam feed
        cv2.imshow('Anti-Theft Detection', display_frame)

        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
