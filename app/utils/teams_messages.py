import pymsteams


class TeamsMessenger:

    @staticmethod
    def send_message(title, text, webhook):
        # You must create the connectorcard object with the Microsoft Webhook URL
        team_message = pymsteams.connectorcard(webhook)

        team_message.title(title)
        # Add text to the message.
        team_message.text(text)

        # send the message.
        team_message.send()
