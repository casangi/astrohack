import sys

pyfile_name = sys.argv[1]
config_name = sys.argv[2]

def parse_file(fname):
    global_dict = {}
    with open(fname, 'r') as pyfile:
        function_head = []
        in_function = False
        n_3quotes = 0
        func_name = ''
        for line in pyfile:
            if in_function:
                if '"""' in line or "'''" in line:
                    n_3quotes += 1
                function_head.append(line)
                if n_3quotes >= 2:
                    in_function = False
                    n_3quotes = 0
                    global_dict[func_name] = parse_function(function_head)
                    function_head = []
            else:
                wrds = line.split()
                if len(wrds) > 0:
                    if wrds[0] == 'def':
                        in_function = True
                        func_name = wrds[1].replace('(','')
                        function_head.append(line)
            
    
    return global_dict

def parse_function(header):
    param_list = get_param_names(header)
    func_dict = {}
    for param in param_list:
        func_dict[param] = get_param_dict(param, header)

    return func_dict

def get_param_names(header):
    lestr = ''
    for line in header:
        lestr += line.replace('\n','')
        if ':' in line:
            break

    param_list = lestr.split('(')[1].split(')')[0].split(',')
    for ipar in range(len(param_list)):
        param_list[ipar] = param_list[ipar].split('=')[0].strip()
    if 'self' in param_list:
        param_list.remove('self')
    return param_list

def get_param_dict(param, header):
    missing = '_###MISSING###_'
    param_dict = {'allowed': missing}
    
    typstr = ''
    required = True
    for line in header:
        if f'{param}=' in line:
            required = False
        if f':type {param}:' in line:
            typstr = line.replace('\n','').strip()
    param_dict['required'] = required

    types = []
    if typstr != '':
        delimiters = [" ", "or", ",", "optional"]
        types = typstr.split(':')[2]
        for delimiter in delimiters:
             types = " ".join(types.split(delimiter))
        types = types.split()

    edit_list_entry('bool', types, 'boolean')
    edit_list_entry('numpy.ndarray', types, 'ndarray')
    if 'dtype' in types:
        types.remove('dtype')
        
    if len(types) == 0:
        param_dict['type'] = missing
    else:
        param_dict['type'] = types

    if 'list' in types or 'ndarray' in types or 'tuple' in types:
        param_dict['struct_type'] = []
        param_dict['minlength'] = missing
        param_dict['maxlength'] = missing
        for typ in types:
            if not typ in ['list', 'ndarray', 'tuple']:
                param_dict['struct_type'].append(typ)

    if 'int' in types or 'float' in types:
        param_dict['min'] = missing
        param_dict['max'] = missing

    
    return param_dict

def edit_list_entry(key, lista, replacement):
    if key in lista:
        lista[lista.index(key)] = replacement
    

def write_json(global_dict, config_name):
    op = '{\n'
    cl = '}\n'
    clc = '},\n'
    spc = '    '
    with open(config_name, "w") as outfile:
        outfile.write(op)
        for func in global_dict.keys():
            outfile.write(f'{spc}"{func}":{op}')
            for param in global_dict[func].keys():
                outfile.write(f'{2*spc}"{param}":{op}')
                for key in global_dict[func][param].keys():
                    outfile.write(write_key(key, global_dict[func][param][key]))
                outfile.write(f'{2*spc}{clc}')
            outfile.write(f'{spc}{clc}')
        outfile.write(cl)

def write_key(key, value):
    spc = '    '
    base = f'{3*spc}"{key}":'
    if isinstance(value, list):
        extra = '['
        for ival in range(len(value)):
            extra += f'"{value[ival]}"'
            if ival != len(value)-1:
                extra += ', '
        extra += ']'
        
        return base+f' {extra}\n'
    if isinstance(value, bool):
        return base+f' {str(value).lower()}\n'
    else:
        return base+f' "{value}"\n'

global_dict = parse_file(pyfile_name)
write_json(global_dict, config_name)



