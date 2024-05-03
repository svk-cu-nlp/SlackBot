# Slack_Chatbot

This guide provides step-by-step instructions to integrate a chatbot with Slack using ngrok for local testing.

## Slack Chatbot Creation
Go to [Slack APP Dashboard](https://api.slack.com/apps/) Follow the documentation to create the App.

## Setting up Python

1. **Install Dependencies**: Install the dependencies from `requirements.txt` using the following command:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Application**: Start the application by running `app.py`.
   ```
   python app.py
   ```

2. **Run ngrok**: Run ngrok to expose your local server to the internet. Use the following command:
   ```
   ./ngrok http 5000
   ```

3. **Copy Public URL**: Copy the public URL provided by ngrok.

4. **Paste into Slack API Dashboard**: Paste the public URL into the Slack API dashboard's Event Subscriptions.

5. **Reinstall Slack Application**:
   - Go to the Basic Information tab of your Slack App.
   - Reinstall the Slack Application in the workspace by following the steps provided.
   - In the Slack chat dashboard, invite the app (created chatbot) to your workspace.

6. **Set Permissions**:
   - Go to the OAuth and Permission Tab in your Slack App settings.
   - Allow the following permissions in the Scope section:
     - `app mention`
     - `chat write`
     - `channel history`

