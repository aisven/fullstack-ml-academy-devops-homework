import requests


def get_hello_world() -> requests.Response:
    url_hello_world_local = "http://127.0.0.1:5000/hello_world"
    res = requests.get(url_hello_world_local)
    return res


def main():
    res_hello_world: requests.Response = get_hello_world()
    res_hello_world_str = str(res_hello_world.content.decode(res_hello_world.encoding))
    print(res_hello_world.status_code)
    print(res_hello_world_str)


if __name__ == "__main__":
    main()
