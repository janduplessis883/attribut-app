from researchcrew_ai.utils import time_it
import time



@time_it
def test_function():
    time.sleep(5)
    return "Done"



if __name__ == "__main__":
    test_function()
    pass
