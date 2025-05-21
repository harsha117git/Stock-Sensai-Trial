import os

# Set API keys as environment variables
os.environ["OPENAI_API_KEY"] = "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"
os.environ["ANAKIN_API_KEY"] = "APS-cMnYEWy9wT2DlM5IhWN1RuGRgUmlxMeW"
os.environ["ANTHROPIC_API_KEY"] = "APS-cMnYEWy9wT2DlM5IhWN1RuGRgUmlxMeW"  # Use Anakin API key for Anthropic calls

print("Environment variables set successfully!")
print(f"OPENAI_API_KEY: {os.environ.get('OPENAI_API_KEY')}")
print(f"ANAKIN_API_KEY: {os.environ.get('ANAKIN_API_KEY')}")
print(f"ANTHROPIC_API_KEY: {os.environ.get('ANTHROPIC_API_KEY')}")