""" Main function to run simEngine
"""


def runSimulation(filepath='test_pendulum.adm'):

    with open(filepath) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            print("Line {}: {}".format(cnt, line.strip()))
            line = fp.readline()
            cnt += 1


if __name__ == "__main__":
    runSimulation()