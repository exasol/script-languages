#include "streamingcontainer.h"
#include <iostream>
#include <string>
#include <functional>
#include <map>
#include <sstream>
#include <unistd.h>
#include <ext/stdio_filebuf.h>
#include <fstream>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <mutex>
#include <type_traits>

using namespace SWIGVMContainers;
using namespace std;


bool mexec(const std::string& cmd_, std::string& result) {
    char buffer[128];
    std::stringstream cmd;
    cmd << "ulimit -v 500000; ";
    cmd << cmd_ << " 2>&1";

    FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) {
        result = "Cannot start command `" + cmd.str() + "`";
        return false;
    }
    while (!feof(pipe)) {
        if (fgets(buffer, 128, pipe) != NULL) {
            result += buffer;
        }
    }
    int s = pclose(pipe);
    if (s == -1)
    {
        return false;
    }
    if (WEXITSTATUS(s))
    {
        return false;
    }
    return true;
}


StreamingVM::StreamingVM(bool checkOnly)
    : meta(), inp(), outp(&inp),
      csvPrinters{
{UNSUPPORTED,[](ostream&, unsigned int) -> void {
    throw SWIGVM::exception("Trying to serialize UNSUPPORTED datatype to CSV");
}},
{DOUBLE,[&](ostream&str, unsigned int col) {str << inp.getDouble(col);}},
{INT32,[&](ostream&str, unsigned int col) {str << inp.getInt32(col);}},
{INT64,[&](ostream&str, unsigned int col) {str << inp.getInt64(col);}},
{NUMERIC,[&](ostream&str, unsigned int col) {str << inp.getNumeric(col);}},
{TIMESTAMP,[&](ostream&str, unsigned int col) {str << inp.getTimestamp(col);}},
{DATE,[&](ostream&str, unsigned int col) {str << inp.getDate(col);}},
{STRING,[&](ostream&str, unsigned int col) {
    //string o = inp.getString(col);
    //std::cerr << "Sending Input: [" << o << "] for column " << col << std::endl;
    str << inp.getString(col);
}},
{BOOLEAN,[&](ostream&str, unsigned int col) {str << (inp.getBoolean(col)?"true":"false");}},
{INTERVALYM,[&](ostream&, unsigned int) {
    throw SWIGVM::exception("Trying to serialize INTERVALYM datatype to CSV");
}},
{INTERVALDS,[&](ostream&, unsigned int) {
    throw SWIGVM::exception("Trying to serialize INTERVALDS datatype to CSV");
}},
{GEOMETRY,[&](ostream&, unsigned int) {
    throw SWIGVM::exception("Trying to serialize GEOMETRY datatype to CSV");
}}
},


csvReaders{

//    {UNSUPPORTED,[](ostream&, unsigned int) -> void {
//        throw SWIGVM::exception("Trying to serialize UNSUPPORTED datatype to CSV");
//    }},
//    {DOUBLE,[&](ostream&str, unsigned int col) {str << inp.getDouble(col);}},
//    {INT32,[&](ostream&str, unsigned int col) {str << inp.getInt32(col);}},
//    {INT64,[&](ostream&str, unsigned int col) {str << inp.getInt64(col);}},
//    {NUMERIC,[&](ostream&str, unsigned int col) {str << inp.getNumeric(col);}},
//    {TIMESTAMP,[&](ostream&str, unsigned int col) {str << inp.getTimestamp(col);}},
//    {DATE,[&](ostream&str, unsigned int col) {str << inp.getDate(col);}},
    {STRING,[&](string&str, unsigned int col) {
            std::cerr << "******* SETTING RESULT: [" << str << "] for column " << col << std::endl;
            outp.setString(col,str.c_str(),-1);}}
//    ,
//    {BOOLEAN,[&](ostream&str, unsigned int col) {str << (inp.getBoolean(col)?"true":"false");}},
//    {INTERVALYM,[&](ostream&, unsigned int) {
//        throw SWIGVM::exception("Trying to serialize INTERVALYM datatype to CSV");
//    }},
//    {INTERVALDS,[&](ostream&, unsigned int) {
//        throw SWIGVM::exception("Trying to serialize INTERVALDS datatype to CSV");
//    }},
//    {GEOMETRY,[&](ostream&, unsigned int) {
//        throw SWIGVM::exception("Trying to serialize GEOMETRY datatype to CSV");
//    }}

}

{


    if (meta.inputType()  != MULTIPLE
            || meta.outputType() != MULTIPLE)
    {
        throw SWIGVM::exception("STREAMING language container only support SET-EMITS UDFs");
    }


    ofstream f("/tmp/program");
    f << meta.scriptCode();
    f.close();

    string res;
    if (!mexec("chmod +x /tmp/program",res)) {
        std::stringstream msg;
        msg << "Error when making script code executable: " << res;
        throw SWIGVM::exception(msg.str().c_str());
    }

    readBuffer.reserve(1024 * 1024 * 2);
}

void StreamingVM::shutdown()
{

}

void StreamingVM::inputToCSV(ostream&str)
{
    if (meta.inputColumnCount()>0) {
        csvPrinters[meta.inputColumnType(0)](str,0);
        for (unsigned int col = 1; col<meta.inputColumnCount(); col++) {           
            str << ",";
            csvPrinters[meta.inputColumnType(col)](str,col);
        }
    }
    str<<"\n";
}


bool StreamingVM::readCSVValue(istream&str, unsigned int column)
{
    bool lastColumn = ((column+1) == meta.outputColumnCount());
    if (str.eof()) {
        throw SWIGVM::exception("readCSVValue after eof");
    }
    readBuffer.clear();
    size_t numRead = 0;
    while (true) {
        int current = str.get();
        if (current == string::traits_type::eof()) {
            if (lastColumn) { // ok, last column is empty value
                break;
            } else {
                throw SWIGVM::exception("eof before last column");
            }
        } else
        if (current == '\n') {
            if (lastColumn) { // ok, last column is empty value
                break;
            } else {
                throw SWIGVM::exception("end of line before last column");
            }
        } else
        if (current == ',') {
            if (lastColumn) {
                throw SWIGVM::exception("delimiter after last column");
            }
            break;
        } else {
            readBuffer += current;
            numRead++;
        }
    }
//    if (str.eof() && numRead == 0) {
//        return false;
//    }
    readBuffer[numRead] = '\0';
    return true;
}

bool StreamingVM::CSVToOutput(istream&str)
{
    // read stream until line is finished or eof
    if (!readCSVValue(str, 0)) {
        return false;
    }
    if (outp.rowsEmited() > 0) {
        outp.next(); // make room for the data unless it is the first emitted row
    }
    csvReaders[meta.outputColumnType(0)](readBuffer,0);
    for (unsigned int col = 1; col<meta.outputColumnCount(); col++) {
        readCSVValue(str, col);
        csvReaders[meta.outputColumnType(col)](readBuffer,col);
    }
    return !str.eof();
}

void handler(int s) {
printf("Caught SIGPIPE\n");
}

bool StreamingVM::run()
{
    signal(SIGPIPE, handler);
    try {
#define PIPE_READ_END 0
#define PIPE_WRITE_END 1


    int pproc_in[2];
    pipe(pproc_in);

    int pproc_out[2];
    pipe(pproc_out);

    int pproc_error[2];
    pipe(pproc_error);


    int pproc_pid = fork();
    if (pproc_pid == 0) {

        close(pproc_in[PIPE_WRITE_END]);
        close(pproc_out[PIPE_READ_END]);
        close(pproc_error[PIPE_READ_END]);

        int res = dup2(pproc_in[PIPE_READ_END], 0);
        if (res < 0) {perror("dup2-1"); exit(1);}
        close(pproc_in[PIPE_READ_END]);

        res = dup2(pproc_out[PIPE_WRITE_END], 1);
        if (res < 0) {perror("dup2-2"); exit(1);}
        close(pproc_out[PIPE_WRITE_END]);

        res = dup2(pproc_error[PIPE_WRITE_END], 2);
        if (res < 0) {perror("dup2-3");exit(1);}
        close(pproc_error[PIPE_WRITE_END]);

        char* const argv[]{(char*)"program", nullptr};
        if (execvp("/tmp/program", argv) == -1) {
            perror("execvp");
            exit(1);
        }
    }

    // I am parent here
    close(pproc_in[PIPE_READ_END]);
    close(pproc_out[PIPE_WRITE_END]);
    close(pproc_error[PIPE_WRITE_END]);

    __gnu_cxx::stdio_filebuf<char> outFilebuf(pproc_in[PIPE_WRITE_END], std::ios::out);
    ostream os(&outFilebuf);

    __gnu_cxx::stdio_filebuf<char> inFilebuf(pproc_out[PIPE_READ_END], std::ios::in);
    istream is(&inFilebuf);

    __gnu_cxx::stdio_filebuf<char> errorFilebuf(pproc_error[PIPE_READ_END], std::ios::in);
    istream es(&errorFilebuf);

    thread t1([&]()
    {
        try {
            do {
                inputToCSV(os);
            } while (inp.next());
            os.flush();
            close(pproc_in[PIPE_WRITE_END]);
        } catch (std::exception& err) {
            lock_guard<mutex> lock(exception_msg_mtx);
            exception_msg = err.what();
        }
    });

    thread t2([&]()
    {
        try {
            while (!is.eof()) {
                if (!CSVToOutput(is)) {
                    break;
                }
            }
            outp.flush();
        } catch (std::exception& err) {
            lock_guard<mutex> lock(exception_msg_mtx);
            exception_msg = err.what();
        }
    });

    t1.join();
    t2.join();


    close(pproc_out[PIPE_READ_END]);

    if (exception_msg.size() == 0) {
        readBuffer.clear();  // error messages from the subprocess?
        while (readBuffer.size()<1000) {
            int current = es.get();
            if (current == string::traits_type::eof()) {
                break;
            }
            readBuffer += current;
        }

        if (!es.eof() && readBuffer.size()>0) {
            readBuffer += "...";
        }
    }
    close(pproc_error[PIPE_READ_END]);
    if (waitpid(pproc_pid,NULL,0) < 0) {
        perror("waitpid");
        exit(EXIT_FAILURE);
    }

    if (exception_msg.size() > 0) {
        throw SWIGVM::exception(exception_msg.c_str());
    }

    if (readBuffer.size()>0) {
        string error_msg = "Error output during script execution:\n"+readBuffer;
        throw SWIGVM::exception(error_msg.c_str());
    }


    } catch (std::exception& ex) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = ex.what();
    }
    return true;  // done
}

const char* StreamingVM::singleCall(single_call_function_id_e fn, const ExecutionGraph::ScriptDTO& args)
{
    throw SWIGVM::exception("singleCall not supported for the STREAMING language container");
}