namespace oneStorage {
    class oneFile{
        public:
            oneFile();
            ~oneFile();

            int    read(const std::string &filename, std::string *result);
            size_t get_file_size(const std::string &filename);            
            void   list_files(const std::string &path, std::vector<std::string> *filenames);
    };
}