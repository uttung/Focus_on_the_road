# Download the helper library from https://www.twilio.com/docs/python/install
from twilio.rest import Client


# Your Account Sid and Auth Token from twilio.com/console
# DANGER! This is insecure. See http://twil.io/secure
account_sid = 'AC4d82f1839e59f6008ec55f3ee1fa2f79'
auth_token = 'b216b65ba998187d6247a06391a942a2'
client = Client(account_sid, auth_token)

message = client.messages.create(
    from_='whatsapp:+14155238886',
    body='Apoorv\'s eyes have been closed for 45 seconds and his trip is still going on!',
    status_callback='http://postb.in/1234abcd',
    to='whatsapp:+919381872407'
)

message = client.messages.create(
    from_='whatsapp:+14155238886',
    body='Apoorv\'s location',
    status_callback='http://postb.in/1234abcd',
    persistent_action=['geo:28.468000,77.091003|Essel Towers'],
    to='whatsapp:+919381872407'
)

print(message.sid)
