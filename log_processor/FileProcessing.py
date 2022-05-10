from LogFileStructure import *


class LogFile:
    def __init__(self, filename):
        self.filename = filename
        self.log_data = LogDict
        self.logger_msgs = LogDict.keys()

    def extract_data(self):
        log_file = open(self.filename, 'r')
        for Line in log_file:
            line = Line.rstrip().split(',')

            if line[0] in self.logger_msgs:
                for Field in self.log_data[line[0]].keys():
                    line_index = self.log_data[line[0]][Field][0]
                    try:
                        if line[line_index] == 'NaN':
                            pass
                        else:
                            self.log_data[line[0]][Field].append(float(line[line_index]))
                    except ValueError:
                        self.log_data[line[0]][Field].append(line[line_index])

                    except IndexError as e:
                        print(e)
                        print(self.log_data[line[0]][Field], line, line_index)


            else:
                pass
        log_file.close()
        return self.log_data

    def get_fields(self, msg):
        """
        This function tells you the Fields of a Logger Message:
        Example:
        Fields('BAT')
        Output will be:
           ['Volt','VoltR','Curr','CurrTot','Temp','Res']
        Now, you know what Fields you could plot against the variable Time
        """
        return list(self.log_data[msg].keys())


if __name__ == '__main__':
    file_name = 'tf.log'
    lf = LogFile(file_name)
    data = lf.extract_data()
    print(list(data.keys()))
    print(lf.get_fields(list(data.keys())[0]))
