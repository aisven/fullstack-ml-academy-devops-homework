import requests


def get_hello_world() -> requests.Response:
    url_hello_world_local = "http://127.0.0.1:5000/hello_world"
    res = requests.get(url_hello_world_local)
    return res


def get_predict() -> requests.Response:
    url_predict_local = "http://127.0.0.1:5000/predict?zylinder=6&ps=133&gewicht=3410&beschleunigung=15.8&baujahr=78"
    res = requests.get(url_predict_local)
    return res


def main():
    res_hello_world: requests.Response = get_hello_world()
    print(res_hello_world.status_code)
    res_hello_world_str = str(res_hello_world.content.decode(res_hello_world.encoding))
    print(res_hello_world_str)

    res_predict: requests.Response = get_predict()
    print(res_predict.status_code)
    res_predict_str = str(res_predict.content.decode(res_predict.encoding))
    print(res_predict_str)


if __name__ == "__main__":
    main()
