import os
import json
import csv
import requests
from typing import List, Dict, Optional
from datetime import datetime

LOCAL_HOST = "http://localhost:1234"

class ChatAssistant:
    def __init__(self):
        self.chat_history: List[Dict] = []
        self.session_start = datetime.now()
        self.auto_save = False
        self.auto_save_path = 'auto_save_chat.json'

    def get_models(self) -> List[str]:
        response = requests.get(f'{LOCAL_HOST}/v1/models')
        return [m['id'] for m in response.json()['data']]

    def chat(self, message: str, model: str = 'lawma') -> str:
        if message.lower() in ['exit', 'quit', 'end', 'stop', 'bye']:
            return self.end_session()

        url = f'{LOCAL_HOST}/v1/chat/completions'
        headers = {'Content-Type': 'application/json'}

        messages = self.chat_history.copy()
        messages.append({'role': 'user', 'content': message})

        data = {
            'model': model,
            'messages': messages,
            'temperature': 0.7,
            'max_tokens': 2000
        }

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        reply = response.json()['choices'][0]['message']['content']
        self.chat_history.append({'role': 'assistant', 'content': reply})

        if self.auto_save:
            self.save_history(self.auto_save_path)
        return reply

    def end_session(self) -> str:
        duration = datetime.now() - self.session_start
        stats = self.get_session_stats()
        self.clear_history()
        return f"Session ended. Duration: {duration}. Stats: {stats}"

    def clear_history(self):
        self.chat_history = []

    def save_history(self, file_path: str, format: str = 'json'):
        if format == 'json':
            with open(file_path, 'w') as f:
                json.dump(self.chat_history, f, indent=2)
        elif format == 'csv':
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Role', 'Content', 'Timestamp'])
                for message in self.chat_history:
                    writer.writerow([message['role'], message['content'], message.get('timestamp', '')])
        elif format == 'txt':
            with open(file_path, 'w') as f:
                for message in self.chat_history:
                    f.write(f"{message['role']}: {message['content']}\n")

    def load_history(self, file_path: str, format: str = 'json'):
        if format == 'json':
            with open(file_path, 'r') as f:
                self.chat_history = json.load(f)
        elif format == 'csv':
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                self.chat_history = [dict(row) for row in reader]
        elif format == 'txt':
            with open(file_path, 'r') as f:
                lines = f.readlines()
                self.chat_history = []
                for line in lines:
                    if ': ' in line:
                        role, content = line.split(': ', 1)
                        self.chat_history.append({'role': role, 'content': content.strip()})

    def search_history(self, query: str) -> List[Dict]:
        return [msg for msg in self.chat_history if query.lower() in msg['content'].lower()]

    def filter_history(self, role: Optional[str] = None, contains: Optional[str] = None) -> List[Dict]:
        filtered = self.chat_history
        if role:
            filtered = [msg for msg in filtered if msg['role'] == role]
        if contains:
            filtered = [msg for msg in filtered if contains.lower() in msg['content'].lower()]
        return filtered

    def get_session_stats(self) -> Dict:
        user_messages = len([msg for msg in self.chat_history if msg['role'] == 'user'])
        assistant_messages = len([msg for msg in self.chat_history if msg['role'] == 'assistant'])
        total_words = sum(len(msg['content'].split()) for msg in self.chat_history)
        return {
            'total_messages': len(self.chat_history),
            'user_messages': user_messages,
            'assistant_messages': assistant_messages,
            'total_words': total_words
        }

    def enable_auto_save(self, file_path: str = 'auto_save_chat.json'):
        self.auto_save = True
        self.auto_save_path = file_path

    def disable_auto_save(self):
        self.auto_save = False


if __name__ == '__main__':
    assistant = ChatAssistant()
    assistant.enable_auto_save()
    print('Welcome to the Chat Assistant! Type \'help\' for commands.')
    while True:
        try:
            message = input('You: ')
            if message.lower() == 'help':
                print('Commands:\n'
                      '  exit/quit/end/stop/bye - End session\n'
                      '  stats - Show session statistics\n'
                      '  clear - Clear chat history\n'
                      '  save <filename> - Save history\n'
                      '  load <filename> - Load history\n'
                      '  search <query> - Search history\n'
                      '  filter <role> <query> - Filter history')
                continue
            reply = assistant.chat(message)
            print(f'Assistant: {reply}')
        except KeyboardInterrupt:
            print(assistant.end_session())
            break
