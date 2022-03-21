import sys

def print_secret(secret :str):
    if secret == "***":
        print("Oh my god!!!")
    print(f"Secret is: {secret}")


for secret in sys.argv[1:]:
    print_secret(secret)