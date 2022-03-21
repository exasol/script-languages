import sys

def print_secret(secret :str):
    if secret == "***":
        print("Oh my god!!!")
    print(f"Secret is: {secret[0]}{secret[1:-1]}{secret[-1]}")


for secret in sys.argv[1:]:
    print_secret(secret)