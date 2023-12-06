import argparse

opn = '{\n'
cls = '}\n'
clsc = '},\n'

parser = argparse.ArgumentParser(description="Parse a python file to produce a skeleton param json file for auror")
parser.add_argument('python_file', type=str, help='Python file to be parsed')
parser.add_argument('json_file', type=str, help='Name of the output json file')
parser.add_argument('-l', '--tab_length', type=int, default=4, required=False, help='Name of the output json file')
args = parser.parse_args()


def edit_list_entry(key, lista, replacement):
    if key in lista:
        lista[lista.index(key)] = replacement


class Param:
    missing = '_###MISSING###_'

    def __init__(self, paramstr):
        param_wrds = paramstr.split('=')
        self.required = len(param_wrds) == 1
        self.name = param_wrds[0].strip()
        self.dicio = {}
        return

    def build_dict(self, header):
        self.dicio = {'allowed': self.missing,
                      'nullable': self.missing,
                      'required': self.required}
    
        typstr = ''
        for line in header:
            if f':type {self.name}:' in line:
                typstr = line.replace('\n', '').strip()
                break

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
            self.dicio['type'] = self.missing
        else:
            self.dicio['type'] = types

        if 'list' in types or 'ndarray' in types or 'tuple' in types:
            self.dicio['struct_type'] = []
            self.dicio['minlength'] = self.missing
            self.dicio['maxlength'] = self.missing
            for typ in types:
                if typ not in ['list', 'ndarray', 'tuple']:
                    self.dicio['struct_type'].append(typ)

        if 'int' in types or 'float' in types:
            self.dicio['min'] = self.missing
            self.dicio['max'] = self.missing

    def export_to_json(self, lvl, spc, op, clc):
        outstr = f'{lvl*spc}"{self.name}":{op}'
        for key in self.dicio.keys():
            outstr += self.write_key(key, lvl+1, spc)
        outstr += f'{lvl*spc}{clc}'        
        return outstr

    def write_key(self, key, lvl, spc):
        value = self.dicio[key]
        base = f'{lvl*spc}"{key}":'
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
    

class Function:
    def __init__(self, name, header):
        self.param_list = []
        self.name = name
        self.header = header
        self._build_param_list()
        return

    def _build_param_list(self):
        self._get_param_list()
        for param in self.param_list:
            param.build_dict(self.header)

    def _get_param_list(self):
        lestr = ''
        for line in self.header:
            lestr += line.replace('\n', '')
            if ':' in line:
                break
        param_str_list = lestr.split('(')[1].split(')')[0].split(',')

        for parstr in param_str_list:
            if 'self' in parstr:
                continue
            else:
                self.param_list.append(Param(parstr))

    def export_to_json(self, lvl, spc, op, clc):
        outstr = f'{lvl*spc}"{self.name}":{op}'
        for param in self.param_list:
            outstr += param.export_to_json(lvl+1, spc, op, clc)
        outstr += f'{spc}{clc}'
        return outstr


class PythonFile:
    def __init__(self, file_name):
        self.file_name = file_name
        self.function_list = []
        self._parse()

    def _parse(self):
        with open(self.file_name, 'r') as pyfile:
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
                        self.function_list.append(Function(func_name, function_head))
                        function_head = []
                else:
                    wrds = line.split()
                    if len(wrds) > 0:
                        if wrds[0] == 'def':
                            in_function = True
                            func_name = wrds[1].replace('(', '')
                            function_head.append(line)

    def export_to_json(self, json_file, spc, op, cl, clc):
        with open(json_file, "w") as outfile:
            outfile.write(op)
            for func in self.function_list:
                outfile.write(func.export_to_json(1, spc, op, clc))
            outfile.write(cl)


mypyfile = PythonFile(args.python_file)
mypyfile.export_to_json(args.json_file, args.tab_length*' ', opn, cls, clsc)
