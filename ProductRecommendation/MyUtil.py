import subprocess,os

# can use this func to run some hdfs command 
def run_cmd(args_list):
        print('Running system command: {0}'.format(' '.join(args_list)))
        # the pipe of running command
        proc = subprocess.Popen(args_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # s_output is the output result if the command has ,s_err is the error info when command can't success
        s_output, s_err = proc.communicate()

        # s_return is the status code to determine the command execution status
        s_return =  proc.returncode

        return s_return, s_output, s_err


def getResultFileName(rootPath, fileName):
	return os.path.join("hdfs://{0}".format(rootPath), fileName)