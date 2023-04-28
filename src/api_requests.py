import requests


def get_hello() -> requests.Response:
    url_hello_local = "http://127.0.0.1:5000"
    res = requests.get(url_hello_local)
    return res


def main():
    res_hello: requests.Response = get_hello()
    res_hello_str = str(res_hello.content.decode(res_hello.encoding))
    print(res_hello.status_code)
    print(res_hello_str)


if __name__ == "__main__":
    main()
