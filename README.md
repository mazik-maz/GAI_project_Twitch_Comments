# GAI Project: Twitch Comments

**GAI Project: Twitch Comments** is an application designed to capture, analyze, and visualize real-time chat data from Twitch channels. It enables users to monitor chat activity, perform sentiment analysis, and gain insights from live comments during streaming sessions.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

This project connects to Twitch’s API to capture live chat comments and processes the data to provide useful analytics. By integrating natural language processing and data visualization, it offers a comprehensive view of viewer interactions on Twitch channels, making it valuable for developers, stream moderators, and data enthusiasts looking to understand live audience behavior.

---

## Features

- **Live Chat Capture:** Connects to Twitch’s API to stream real-time chat messages.
- **Sentiment Analysis:** Evaluates the sentiment of chat messages using NLP techniques.
- **Data Visualization:** Generates charts and graphs to highlight chat activity and trends.
- **Modular Design:** Easily extendable with additional features or analytical modules.
- **Configurable Options:** Allows customization of polling intervals, API credentials, and analysis parameters.

---

## Requirements

- **Python:** Version 3.8 or newer.
- **Twitch Developer Account:** To obtain necessary API credentials.
- **Dependencies:** Install required Python packages listed in `requirements.txt`, such as:
  - `requests`
  - `pandas`
  - `matplotlib`
  - `nltk` (or any alternative NLP library)

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/mazik-maz/GAI_project_Twitch_Comments.git
   cd GAI_project_Twitch_Comments
