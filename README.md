# pacman

ECS 170 Group 9.

## Prerequisites

- Python 3.8 or later
- pip (usually included with Python)

## Create and activate a virtual environment (Linux / macOS)

1. Create the virtual environment:

```
python3 -m venv .venv
```

2. Activate it:

```
source .venv/bin/activate
```

On Windows (PowerShell):

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## Install dependencies with pip

After activating the virtual environment, upgrade pip and install the project's dependencies:

```
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Deactivate the virtual environment

When you're done working:

```
deactivate
```

## If `requirements.txt` is missing

You can create it from an active environment:

```
pip freeze > requirements.txt
```

---

These steps set up an isolated Python environment and install all dependencies using pip.
