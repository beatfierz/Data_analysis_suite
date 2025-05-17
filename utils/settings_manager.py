
import os

class SettingsManager:
    def __init__(self, file_path="settings.ini"):
        self.file_path = file_path
        self.settings = {}
        self.load()

    def load(self):
        if not os.path.exists(self.file_path):
            return
        with open(self.file_path, 'r') as f:
            for line in f:
                if ',' in line:
                    key, value = line.strip().split(',', 1)
                    self.settings[key.strip()] = value.strip()

    def save(self):
        with open(self.file_path, 'w') as f:
            for key, value in self.settings.items():
                f.write(f"{key}, {value}\n")

    def get(self, key, default=None):
        return self.settings.get(key, default)

    def set(self, key, value):
        self.settings[key] = str(value)
        self.save()
