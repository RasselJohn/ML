def create_answer_file(filename, write_data):
    open(f'../output/{filename}', 'w').write(write_data)
