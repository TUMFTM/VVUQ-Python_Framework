"""
This module is responsible for the remote control of CarMaker.

It includes one class CarMakerRemote for the CarMaker control. See details in its own documentation.

Contact person: Stefan Riedmaier
Creation date: 24.05.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --
import socket
import subprocess
import os
import platform
import sys
import shutil
import time

# -- third-party imports --
import xarray as xr

# -- custom imports --


# -- CONSTANTS ---------------------------------------------------------------------------------------------------------
CREATE_NEW_PROCESS_GROUP = 0x00000200
DETACHED_PROCESS = 0x00000008


# -- CLASSES -----------------------------------------------------------------------------------------------------------
class CarMakerRemote:
    """
    This class is responsible for the remote control of CarMaker.

    It includes several methods for launching, tcp/ip connection, TestManager control, etc.
    See the respective function documentations for more details.
    """

    def __init__(self, config, domain, instance):
        """
        This method initializes a new class instance.

        :param dict config: configuration dictionary
        :param str domain: type of VVUQ domain
        :param str instance: test instance
        """

        # -- ASSIGN PARAMETERS TO INSTANCE ATTRIBUTES ------------------------------------------------------------------
        self.config = config
        self.domain = domain
        self.instance = instance

        # -- CREATE CONFIG SUB-DICT POINTERS ---------------------------------------------------------------------------
        self.cfgti = self.config['cross_domain'][instance]
        self.cfgqu = self.config['cross_domain']['quantities']

        # -- INSTANTIATE FURTHER INSTANCE ATTRIBUTES -------------------------------------------------------------------
        self.socket = None

    def launch_carmaker(self, detached_flag=True):
        """
        This function starts the CarMaker application with TCP/IP port.

        :param bool detached_flag: flag to indicate whether CarMaker should stay alive or not
        :return:
        """

        kwargs = {}
        if detached_flag:
            # keep the CarMaker subprocess alive after the python parent exits (see detached, new session, daemon)
            # https://stackoverflow.com/questions/13243807/popen-waiting-for-child-process-even-when-the-immediate-child-has-terminated/13256908#13256908
            if platform.system() == 'Windows':
                # Windows
                kwargs.update(creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP)
            elif sys.version_info < (3, 2):
                # Unix Python 3.1-
                kwargs.update(preexec_fn=os.setsid)
            else:
                # Unix Python 3.2+
                kwargs.update(start_new_session=True)

        # launch CarMaker
        subprocess.run([os.path.normpath(self.cfgti['exe']), '-cmdport', str(self.cfgti['port']), '-user', 'na',
                        self.cfgti['project']], **kwargs)

    def configure_carmaker_initially(self):
        """
        This function initialies CarMaker with the desired settings.

        It includes constant settings such as the CarMaker project or testrun.

        :return:
        """
        quantity_string = ' '.join(self.cfgqu['quantities_name_list'])

        # Load Testrun
        # Delete all quantities from the OutputQuantities file
        # Add the quantities of interest
        # Save all results to disk (not just time span or collect_only)
        # Specify result file path and naming of the erg files
        tcl_command = 'LoadTestRun ' + self.cfgti['testrun'] + '\r' + \
                      'OutQuantsDelAll\r' + \
                      'OutQuantsAdd ' + quantity_string + '\r' + \
                      'SaveMode save\r'
        expected_answer = '\r\n' + 'O0\r\n\r\n' + 'O0\r\n\r\n' + '\r\n'

        self.send_tcpip_message(tcl_command)
        self.receive_tcpip_message(expected_answer)

    def configure_carmaker_before_simulation(self, parameter_values, erg_path, parameter_name_list):
        """
        This function configures CarMaker before each execution of the simulation.

        It includes scenario specific settings such as the parameter combinations.

        :param np.ndarray parameter_values: 1d array with parameter values (same order as name list)
        :param str erg_path: path where the erg file should be stored
        :param list[str] parameter_name_list: list with the parameter names
        :return:
        """

        if len(parameter_values) != len(parameter_name_list):
            raise

        # set the path where to store the erg file
        tcl_command = 'SetResultFName ' + erg_path + '\r'
        # -- set the parameter values in CarMaker
        for (name, value) in zip(parameter_name_list, parameter_values):
            # distinguish between key values and named values starting with a $ in CarMaker
            if name[0] == "$":
                single_tcl_command = 'NamedValue set ' + name[1:] + ' ' + str(value) + '\r'
            else:
                single_tcl_command = 'KeyValue set ' + name + ' ' + str(value) + '\r'
            tcl_command = tcl_command + single_tcl_command
        expected_answer = '\r\n' + '\r\n' * len(parameter_values)

        self.send_tcpip_message(tcl_command)
        self.receive_tcpip_message(expected_answer)

    def execute_simulation(self):
        """
        This function executes the simulation run and waits until it is finished.

        :return:
        """

        # Start the simulation
        # Wait until the simulation is running with timeout of 5s
        tcl_command = 'StartSim\r' + \
                      'WaitForStatus running 5000\r'
        expected_value = '\r\n' + 'O0\r\n\r\n'

        self.send_tcpip_message(tcl_command)
        self.receive_tcpip_message(expected_value)

        # Wait until it is terminated (status idle)
        # Restore the the old OutputQuantities file
        tcl_command = 'WaitForStatus idle\r' + \
                      'OutQuantsRestoreDefs\r'
        expected_value = 'O0\r\n\r\n' + 'O0\r\n\r\n'

        self.send_tcpip_message(tcl_command)
        self.receive_tcpip_message(expected_value)

    def execute_testseries(self, scenarios_da, erg_path_list):
        """
        This function configures and executes a whole test series.

        :param xr.DataArray scenarios_da: array with parameter vectors
        :param list erg_path_list: list with erg paths where the results should be stored
        :return:
        """

        # -- CREATE NEW TEST SERIES FILE --
        tcl_command = 'TestMgr new -force\r' + \
                      'TestMgr save testseries.ts' + '\r'
        expected_value = '\r\n' + '\r\n'

        self.send_tcpip_message(tcl_command)
        self.receive_tcpip_message(expected_value)

        # -- FILL THE TEST SERIES FILE --
        ts_path = self.cfgti['project'] + '/Data/TestRun/' + 'testseries.ts'
        with open(ts_path, 'a') as ts:

            # -- write the testrun with its name
            ts.write('Step.1 = TestRun\n')
            ts.write('Step.1.Name = ' + self.cfgti['testrun'] + '\n')

            # -- write the testrun parameters
            # loop trough the columns of the parameter array
            parameter_name_list = scenarios_da.parameters.values.tolist()
            for i in range(scenarios_da.shape[scenarios_da.dims.index('parameters')]):
                # distinguish between key values and named values starting with a $ in CarMaker
                if parameter_name_list[i][0] == "$":
                    ts.write('Step.1.Param.' + str(i) + ' = ' + parameter_name_list[i][1:] + ' NValue' + '\n')
                else:
                    ts.write('Step.1.Param.' + str(i) + ' = ' + parameter_name_list[i] + ' KValue' + '\n')

            # add an additional parameter for the storage of each variation
            ts.write('Step.1.Param.' + str(i+1) + ' = ResultFName CM\n')

            # -- write the testrun variations
            # loop trough the rows of the parameter array
            for i in range(scenarios_da.samples_2d.shape[0]):
                ts.write('Step.1.Var.' + str(i) + '.Name = Variation ' + str(i) + '\n')
                ts.write('Step.1.Var.' + str(i) + '.Param = ' + ' '.join(map(str, scenarios_da.samples_2d[i]))
                         + ' ' + erg_path_list[i] + '\n')

        ts.close()

        # -- ARCHIVE TEST SERIES FILE --
        dst = self.cfgti['result_folder'] + '/' + self.domain
        src = ts_path
        shutil.copyfile(src, dst + '/testseries.ts')

        # -- EXECUTE THE TEST SERIES FILE --
        tcl_command = 'TestMgr load testseries.ts' + '\r' + \
                      'TestMgr start\r'
        expected_value = '\r\n' + '\r\n'

        self.send_tcpip_message(tcl_command)
        self.receive_tcpip_message(expected_value)

        # -- CARMAKER WORKAROUNDS --

        # The CarMaker Docu describes the command 'TestMgr start' as: "Starts the currently selected test series.
        # [...] the command blocks and returns after the test series has finished completely."
        # However, it already returns when the last testrun of the testseries starts, and not finishes.
        # So, as a workaround, we poll the status of the Test Manager until it switches from 'running' to 'idle'.
        tcl_command = 'TestMgr get Status\r'
        expected_value = 'Orunning\r\n\r\n'

        flag = True
        while flag:
            try:
                # poll the status of the Test Manager in a loop during it returns 'running'
                self.send_tcpip_message(tcl_command)
                self.receive_tcpip_message(expected_value)

                # wait for ... seconds so that CarMaker is not overloaded
                time.sleep(1)
            except ValueError as e:
                # if the status switches from 'running' to 'idle', we catch the desired exception! (quick workaround)
                expected_value = 'Oidle\r\n\r\n'
                # if idle, exit the loop, otherwise re-raise the exception
                if e.args[1] == expected_value:
                    flag = False
                else:
                    raise

        # remove the sequence option from the result paths that CarMaker always adds, even if not selected
        self.correct_erg_filenames()

    def correct_erg_filenames(self):
        """
        This function corrects the erg-file names, since the TestManager always adds a sequence number in parallel mode.

        Unfortunately the TestManager always adds _00x.erg to the result file names (sequence option: ?_%s) in parallel
        mode, even if the sequence option is not chosen. So we have to remove it afterwards.

        :return:
        """

        # walk trough all result subdirectories
        root_path = self.cfgti['result_folder'] + '/' + self.domain
        for root, dirs, files in os.walk(root_path):
            for name in files:

                # rename the erg and erg.info files
                if name.endswith('erg'):
                    # split the file names based on dots
                    split_list = name.rsplit('.')

                    # if it erroneously contains three parts (name.erg_00x.erg), remove the second part -> name.erg
                    if len(split_list) == 3:
                        new_path = root + '/' + split_list[0] + '.' + split_list[2]

                        # replace the old path with the new one
                        old_path = root + '/' + name
                        os.rename(old_path, new_path)
                elif name.endswith('info'):
                    # split the file names based on dots
                    split_list = name.rsplit('.')

                    # if it erroneously contains four parts (*.erg_00x.erg.info), remove the second part -> *.erg.info
                    if len(split_list) == 4:
                        new_path = root + '/' + split_list[0] + '.' + split_list[2] + '.' + split_list[3]

                        # replace the old path with the new one
                        old_path = root + '/' + name
                        os.rename(old_path, new_path)

    def connect_tcpip_socket(self):
        """
        This function connects the python tpc/ip socket to the CarMaker socket.

        :return:
        """

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.cfgti['ip'], self.cfgti['port']))

    def close_tcpip_socket(self):
        """
        This function closes the tcp/ip socket.

        :return:
        """

        self.socket.close()

    def send_tcpip_message(self, message):
        """
        This function sends an entire tcp/ip message.

        :param str message: the desired message to be transferred
        :return:
        """

        totalsent = 0

        # send the message via tcp/ip until the end of the message is reached
        while totalsent < len(message):
            sent = self.socket.send(message[totalsent:].encode('utf-8'))
            if sent == 0:
                raise RuntimeError("socket connection broken")
            totalsent = totalsent + sent

    def receive_tcpip_message(self, expected_answer):
        """
        This function receives an entire tcp/ip message and compares it with an expected answer.

        It is aimed at the use case where a certain response from the sender is expected when everything has worked.
        Otherwise, the answer will be different.
        The known length of the expected answer is used to determine when the end of the answer is reached.

        It compares the expected answer with the received answer up to the same length.
        It only approves a full match, or aborts otherwise.
        It does not analyse an unknown content from the sender until reaching a delimiter.

        :param str expected_answer: the answer expected from the sender
        :return:
        """

        chunks = ''
        bytes_recd = 0

        # use the known length of the expected answer to determine when the end of the answer is reached
        while bytes_recd < len(expected_answer):
            chunk = self.socket.recv(max(len(expected_answer) - bytes_recd, 200)).decode("utf-8")
            if chunk == '':
                raise RuntimeError("socket connection broken")
            chunks = chunks + chunk
            if chunks not in expected_answer:
                raise ValueError("tcp/ip answer from CarMaker not expected", chunks)
            bytes_recd = bytes_recd + len(chunk)

        return chunks
