#include <fstream>
#include <iostream>
#include <filesystem>
#include <vector>
#include <chrono>
#include <cstring>
#include "io.h"
#include "../../third_party/oneStorage/include/oneapi/onestorage/onefile.h"

namespace fs = std::filesystem;

namespace oneStorage {

size_t oneFile::get_file_size(const std::string &filename)
{
    const auto start_time = std::chrono::high_resolution_clock::now();

    std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);

    const std::chrono::duration<double> total_time =
        std::chrono::high_resolution_clock::now() - start_time;

    /*
    std::cout << "Elapsed time(usec): "
              << "get_file_size = " << total_time.count() * 1e6
              << std::endl;           
    */

    return in.tellg();
}

void oneFile::list_files(const std::string &path, std::vector<std::string> *filenames)
{
    // Iterate over the `std::filesystem::directory_entry` elements using `auto`
    //for (const fs::directory_entry& dir_entry : fs::recursive_directory_iterator("/home/gta/repo/datasets/"))
    for (auto const& dir_entry : fs::recursive_directory_iterator(path))
    {
        //std::cout << dir_entry << '\n';

        if (!fs::is_directory(dir_entry))
        {
            filenames->push_back(fs::absolute( dir_entry.path() ).string());
        }
    }
}

int oneFile::read(const std::string &filename, std::string *result)
{
    int status;
    oneFileDescr_t   onefile_descr;
    oneFileHandle_t  oneFileHandle;
    oneStorageDSET_t dset = new oneStorageDSET_t;

    memset((void *)&onefile_descr, 0, sizeof(oneFileDescr_t));

    oneFileHandle = (void *)&onefile_descr;

    const auto start_time = std::chrono::high_resolution_clock::now();

    // open the dataset, select the proper connector
    status = onestorage_dset_open(filename.c_str(), "uring", dset);

    status = onefile_open(dset, filename.c_str(), oneFileHandle);

    if (status == -1)
    {
        return -1;
    }
    
    const auto num_bytes = this->get_file_size(filename);
    result->resize(num_bytes);

    const std::chrono::duration<double> total_time =
        std::chrono::high_resolution_clock::now() - start_time;
    
    /*
    std::cout << "Elapsed time(usec): "
              << "open file " << total_time.count() * 1e6
              << std::endl;
    */

    long long int read_bytes = 0;
    auto r = 0;

    const auto buffer_ptr = result->data() + read_bytes;
    const auto bytes_to_read = num_bytes - read_bytes;

    r = onefile_read(oneFileHandle, buffer_ptr, bytes_to_read);

    //std::cout << "Total Bytes Read: " << r << std::endl;

    read_bytes += r;
/*
    do {
        const auto buffer_ptr = buffer.data() + read_bytes;
        const auto bytes_to_read = num_bytes - read_bytes;
        r = onefile_read(fd, buffer_ptr, bytes_to_read);
        read_bytes += r;
    } while (r > 0);
*/

    if (read_bytes != num_bytes) {
        std::cerr << "read error "
                  << " read_bytes (read) = " << read_bytes << " num_bytes (fstat) = " << num_bytes
                  << std::endl;
    }

    onefile_close(oneFileHandle);

    return 0;

}


oneFile::oneFile()  {}
oneFile::~oneFile() {}

}

/*
using oneStorage::oneFile;

int main()
{
    std::vector<std::string> filenames;

    size_t file_size;
    oneFile onefile;

    onefile.list_files("/home/gta/repo/datasets/", &filenames);

    for (auto & filename : filenames) {
        file_size = onefile.get_file_size(filename);
        std::cout << filename << ':' << file_size <<'\n';
    }
}
*/