from service import identify_service


if __name__ == '__main__':
    # identify target
    image_path = "test/images/strawberry.jpg"

    identify_result = identify_service.identify(image_path)
    print(identify_result)
