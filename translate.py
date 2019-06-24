import ast
from astor import SourceGenerator
from astor.code_gen import set_precedence, Precedence, pretty_source, pretty_string


symbol_data = {
    'Or': '||',
    'And': '&&',
    'Not': '!',
    'Eq': '==',
    'Gt': '>',
    'GtE': '>=',
    'In': 'in',
    'Is': '==',
    'NotEq': '!=',
    'Lt': '<',
    'LtE': '<=',
    'NotIn': 'not in',
    'IsNot': '!=',
    'UAdd': '+',
    'USub': '-'
}

symbol_data = dict((getattr(ast, x, None), y) for x, y in symbol_data.items())

type_placeholder = '<unknown>'


def get_op_symbol(obj, fmt='%s', symbol_data=symbol_data, type=type):
    """Given an AST node object, returns a string containing the symbol.
    """
    return fmt % symbol_data[type(obj)]


class CXXSourceGenerator(SourceGenerator):
    def __init__(self, *args, **kwargs):
        def quick_visit(node):
            sg = SourceGenerator(*args, **kwargs)
            sg.visit(node)
            return ''.join(sg.result)

        super().__init__(*args, **kwargs)
        self.quick_visit = quick_visit
        self.in_class_name = None
        self.in_class_method = False
        self.self_arg_name = None
        self.include_names = set()
        self.class_members = {}

    def body(self, statements, bracket=True):
        if bracket:
            self.write(' {', self.newline)
        self.indentation += 1
        for stmt in statements:
            self.write(stmt)
            if self.result[-1] != '}':
                self.write(';')
            self.newline(stmt)
        self.indentation -= 1
        if bracket:
            self.write(self.newline, '}')

    def else_body(self, elsewhat):
        if elsewhat:
            self.write(self.newline, 'else')
            self.body(elsewhat)

    def visit_arguments(self, node):
        want_comma = []

        def write_comma():
            if want_comma:
                self.write(', ')
            else:
                want_comma.append(True)

        def loop_args(args, defaults):
            set_precedence(Precedence.Comma, defaults)
            padding = [None] * (len(args) - len(defaults))
            for arg, default in zip(args, padding + defaults):
                self.write(write_comma, arg)
                self.conditional_write('=', default)

        posonlyargs = getattr(node, 'posonlyargs', [])
        offset = 0
        if posonlyargs:
            offset += len(node.defaults) - len(node.args)
            loop_args(posonlyargs, node.defaults[:offset])
            self.write(write_comma, '/')

        if self.in_class_method:
            self.self_arg_name = self.quick_visit(node.args[0])
            args = node.args[1:]
        else:
            args = node.args
        loop_args(args, node.defaults[offset:])
        self.conditional_write(write_comma, '*', node.vararg)

        kwonlyargs = self.get_kwonlyargs(node)
        if kwonlyargs:
            if node.vararg is None:
                self.write(write_comma, '*')
            loop_args(kwonlyargs, node.kw_defaults)
        self.conditional_write(write_comma, '**', node.kwarg)

    def get_result(self):
        self.result = list(self.include_names) + self.result
        self.include_names = []
        if self.result and self.result[-1].strip() != '':
            self.result.append('\n')
        return self.result

    # # Statements

    def visit_Assign(self, node):
        set_precedence(node, node.value, *node.targets)
        for target in node.targets:
            self.write(target, ' = ')
            if type(target) is ast.Attribute:
                self.class_members[target] = node.value
        self.visit(node.value)

    # def visit_AugAssign(self, node):
    #     set_precedence(node, node.value, node.target)
    #     self.statement(node, node.target, get_op_symbol(node.op, ' %s= '),
    #                    node.value)

    # def visit_AnnAssign(self, node):
    #     set_precedence(node, node.target, node.annotation)
    #     set_precedence(Precedence.Comma, node.value)
    #     need_parens = isinstance(node.target, ast.Name) and not node.simple
    #     begin = '(' if need_parens else ''
    #     end = ')' if need_parens else ''
    #     self.statement(node, begin, node.target, end, ': ', node.annotation)
    #     self.conditional_write(' = ', node.value)

    # def visit_ImportFrom(self, node):
    #     self.statement(node, 'from ', node.level * '.',
    #                    node.module or '', ' import ')
    #     self.comma_list(node.names)
    #     # Goofy stuff for Python 2.7 _pyio module
    #     if node.module == '__future__' and 'unicode_literals' in (
    #             x.name for x in node.names):
    #         self.using_unicode_literals = True

    # def visit_Import(self, node):
    #     self.statement(node, 'import ')
    #     self.comma_list(node.names)

    # def visit_Expr(self, node):
    #     self.statement(node)
    #     self.generic_visit(node)

    def visit_FunctionDef(self, node, is_async=False):
        # CXX does not support async
        assert not is_async
        no_self_deco = {'staticmethod', 'classmethod'}
        decorators = set(self.quick_visit(n) for n in node.decorator_list)
        no_self = bool(no_self_deco.intersection(decorators))
        if self.in_class_name and not no_self:
            self.in_class_method = True
        self.newline(extra=1 if self.indentation else 2)
        if node.name == '__init__':
            assert self.in_class_name
            self.statement(node, self.in_class_name, '(')
        else:
            prefix = 'static ' if 'staticmethod' in decorators else ''
            self.statement(node, f'{prefix}void {node.name}', '(')
        self.visit_arguments(node.args)
        self.write(')')
        # self.conditional_write(' ->', self.get_returns(node))
        # print(node.body)
        self.body(node.body)
        if not self.indentation:
            self.newline(extra=2)
        if self.in_class_name and no_self:
            self.in_class_method = False

    # introduced in Python 3.5
    def visit_AsyncFunctionDef(self, node):
        raise NotImplementedError('Async function not supported in C++')

    def visit_ClassDef(self, node):
        self.in_class_name = node.name
        self.decorators(node, 2)
        self.statement(node, 'class %s' % node.name)
        if node.bases:
            self.write(': ')
            self.write(', '.join([f'public {self.quick_visit(base)}' for base in node.bases]))
        decl_names = set(self.quick_visit(k) for k in self.class_members)
        decl_body = [f'{type_placeholder} {name}\n' for name in decl_names]
        print(node.body + [self.newline] * 2 + decl_body)
        self.body(node.body + [self.newline] * 2 + decl_body)
        if not self.indentation:
            self.newline(extra=2)
        self.in_class_name = None
        self.class_members = {}

    def visit_If(self, node):
        set_precedence(node, node.test)
        self.statement(node, 'if (', node.test, ')')
        self.body(node.body)
        while True:
            else_ = node.orelse
            if len(else_) == 1 and isinstance(else_[0], ast.If):
                node = else_[0]
                set_precedence(node, node.test)
                self.write(self.newline, 'else if (', node.test, ')')
                self.body(node.body)
            else:
                self.else_body(else_)
                break

    def visit_For(self, node, is_async=False):
        assert not is_async
        set_precedence(node, node.target)
        self.statement(node, 'for (auto ', node.target, ': ', node.iter, ')')
        self.body_or_else(node)

    # def visit_While(self, node):
    #     set_precedence(node, node.test)
    #     self.statement(node, 'while ', node.test, ':')
    #     self.body_or_else(node)

    # deprecated in Python 3.8
    def visit_NameConstant(self, node):
        if node.value is True:
            self.write('true')
        elif node.value is False:
            self.write('false')
        else:
            self.write(repr(node.value))

    def visit_Print(self, node):
        # XXX: python 2.6 only
        raise NotImplementedError('Python 2 print is not supported')

    # def visit_TryExcept(self, node):
    #     self.statement(node, 'try:')
    #     self.body(node.body)
    #     self.write(*node.handlers)
    #     self.else_body(node.orelse)

    # # new for Python 3.3
    # def visit_Try(self, node):
    #     self.statement(node, 'try:')
    #     self.body(node.body)
    #     self.write(*node.handlers)
    #     self.else_body(node.orelse)
    #     if node.finalbody:
    #         self.statement(node, 'finally:')
    #         self.body(node.finalbody)

    # def visit_ExceptHandler(self, node):
    #     self.statement(node, 'except')
    #     if self.conditional_write(' ', node.type):
    #         self.conditional_write(' as ', node.name)
    #     self.write(':')
    #     self.body(node.body)

    # def visit_TryFinally(self, node):
    #     self.statement(node, 'try:')
    #     self.body(node.body)
    #     self.statement(node, 'finally:')
    #     self.body(node.finalbody)

    # def visit_Exec(self, node):
    #     dicts = node.globals, node.locals
    #     dicts = dicts[::-1] if dicts[0] is None else dicts
    #     self.statement(node, 'exec ', node.body)
    #     self.conditional_write(' in ', dicts[0])
    #     self.conditional_write(', ', dicts[1])

    def visit_Assert(self, node):
        set_precedence(node, node.test, node.msg)
        self.statement(node, 'assert(', node.test)
        self.conditional_write(', ', node.msg)
        self.write(')')

    # def visit_Global(self, node):
    #     self.statement(node, 'global ', ', '.join(node.names))

    # def visit_Nonlocal(self, node):
    #     self.statement(node, 'nonlocal ', ', '.join(node.names))

    def visit_Break(self, node):
        self.statement(node, 'break;')

    def visit_Continue(self, node):
        self.statement(node, 'continue;')

    def visit_Raise(self, node):
        # XXX: Python 2.6 / 3.0 compatibility
        self.statement(node, 'throw')
        if self.conditional_write(' ', self.get_exc(node)):
            self.conditional_write(' from ', node.cause)
        elif self.conditional_write(' ', self.get_type(node)):
            set_precedence(node, node.inst)
            self.conditional_write(', ', node.inst)
            self.conditional_write(', ', node.tback)

    # # Expressions

    def visit_Attribute(self, node):
        if self.in_class_method and type(node.value) is ast.Name and node.value.id == self.self_arg_name:
            # print(f'Replacing self variable {node.value.id} by "this->":')
            self.write('this->', node.attr)
        else:
            self.write(node.value, '.', node.attr)

    # def visit_Call(self, node, len=len):
    #     write = self.write
    #     want_comma = []

    #     def write_comma():
    #         if want_comma:
    #             write(', ')
    #         else:
    #             want_comma.append(True)

    #     args = node.args
    #     keywords = node.keywords
    #     starargs = self.get_starargs(node)
    #     kwargs = self.get_kwargs(node)
    #     numargs = len(args) + len(keywords)
    #     numargs += starargs is not None
    #     numargs += kwargs is not None
    #     p = Precedence.Comma if numargs > 1 else Precedence.call_one_arg
    #     set_precedence(p, *args)
    #     self.visit(node.func)
    #     write('(')
    #     for arg in args:
    #         write(write_comma, arg)

    #     set_precedence(Precedence.Comma, *(x.value for x in keywords))
    #     for keyword in keywords:
    #         # a keyword.arg of None indicates dictionary unpacking
    #         # (Python >= 3.5)
    #         arg = keyword.arg or ''
    #         write(write_comma, arg, '=' if arg else '**', keyword.value)
    #     # 3.5 no longer has these
    #     self.conditional_write(write_comma, '*', starargs)
    #     self.conditional_write(write_comma, '**', kwargs)
    #     write(')')

    # # ast.Constant is new in Python 3.6 and it replaces ast.Bytes,
    # # ast.Ellipsis, ast.NameConstant, ast.Num, ast.Str in Python 3.8
    def visit_Constant(self, node):
        value = node.value

        if isinstance(value, (int, float, complex)):
            with self.delimit(node):
                self._handle_numeric_constant(value)
        elif isinstance(value, str):
            self._handle_string_constant(node, node.value)
        elif value is Ellipsis:
            self.write('...')
        else:
            self.write(repr(value))

    # def visit_JoinedStr(self, node):
    #     self._handle_string_constant(node, None, is_joined=True)

    def _handle_string_constant(self, node, value, is_joined=False):
        # Decide whether is triple quote as comment
        is_comment = False
        if is_comment:
            self.write('/*', value, '*/')
        else:
            super()._handle_string_constant(node, value, is_joined)

    # # deprecated in Python 3.8
    # def visit_Str(self, node):
    #     self._handle_string_constant(node, node.s)

    # # deprecated in Python 3.8
    # def visit_Bytes(self, node):
    #     self.write(repr(node.s))

    # def _handle_numeric_constant(self, value):
    #     x = value

    #     def part(p, imaginary):
    #         # Represent infinity as 1e1000 and NaN as 1e1000-1e1000.
    #         s = 'j' if imaginary else ''
    #         try:
    #             if math.isinf(p):
    #                 if p < 0:
    #                     return '-1e1000' + s
    #                 return '1e1000' + s
    #             if math.isnan(p):
    #                 return '(1e1000%s-1e1000%s)' % (s, s)
    #         except OverflowError:
    #             # math.isinf will raise this when given an integer
    #             # that's too large to convert to a float.
    #             pass
    #         return repr(p) + s

    #     real = part(x.real if isinstance(x, complex) else x, imaginary=False)
    #     if isinstance(x, complex):
    #         imag = part(x.imag, imaginary=True)
    #         if x.real == 0:
    #             s = imag
    #         elif x.imag == 0:
    #             s = '(%s+0j)' % real
    #         else:
    #             # x has nonzero real and imaginary parts.
    #             s = '(%s%s%s)' % (real, ['+', ''][imag.startswith('-')], imag)
    #     else:
    #         s = real
    #     self.write(s)

    # def visit_Num(self, node,
    #               # constants
    #               new=sys.version_info >= (3, 0)):
    #     with self.delimit(node) as delimiters:
    #         self._handle_numeric_constant(node.n)

    #         # We can leave the delimiters handling in visit_Num
    #         # since this is meant to handle a Python 2.x specific
    #         # issue and ast.Constant exists only in 3.6+

    #         # The Python 2.x compiler merges a unary minus
    #         # with a number.  This is a premature optimization
    #         # that we deal with here...
    #         if not new and delimiters.discard:
    #             if not isinstance(node.n, complex) and node.n < 0:
    #                 pow_lhs = Precedence.Pow + 1
    #                 delimiters.discard = delimiters.pp != pow_lhs
    #             else:
    #                 op = self.get__p_op(node)
    #                 delimiters.discard = not isinstance(op, ast.USub)

    # def visit_Tuple(self, node):
    #     with self.delimit(node) as delimiters:
    #         # Two things are special about tuples:
    #         #   1) We cannot discard the enclosing parentheses if empty
    #         #   2) We need the trailing comma if only one item
    #         elts = node.elts
    #         delimiters.discard = delimiters.discard and elts
    #         self.comma_list(elts, len(elts) == 1)

    def visit_List(self, node):
        self.write(f'vector<{type_placeholder}>(')
        with self.delimit('{}'):
            self.comma_list(node.elts)
        self.write(')')

    # def visit_Set(self, node):
    #     if node.elts:
    #         with self.delimit('{}'):
    #             self.comma_list(node.elts)
    #     else:
    #         # If we tried to use "{}" to represent an empty set, it would be
    #         # interpreted as an empty dictionary. We can't use "set()" either
    #         # because the name "set" might be rebound.
    #         self.write('{1}.__class__()')

    # def visit_Dict(self, node):
    #     set_precedence(Precedence.Comma, *node.values)
    #     with self.delimit('{}'):
    #         for idx, (key, value) in enumerate(zip(node.keys, node.values)):
    #             self.write(', ' if idx else '',
    #                        key if key else '',
    #                        ': ' if key else '**', value)

    # def visit_BinOp(self, node):
    #     op, left, right = node.op, node.left, node.right
    #     with self.delimit(node, op) as delimiters:
    #         ispow = isinstance(op, ast.Pow)
    #         p = delimiters.p
    #         set_precedence((Precedence.Pow + 1) if ispow else p, left)
    #         set_precedence(Precedence.PowRHS if ispow else (p + 1), right)
    #         self.write(left, get_op_symbol(op, ' %s '), right)

    def visit_BoolOp(self, node):
        with self.delimit(node, node.op) as delimiters:
            op = get_op_symbol(node.op, ' %s ')
            set_precedence(delimiters.p + 1, *node.values)
            for idx, value in enumerate(node.values):
                self.write(idx and op or '', value)

    def visit_Compare(self, node):
        with self.delimit(node, node.ops[0]) as delimiters:
            set_precedence(delimiters.p + 1, node.left, *node.comparators)
            self.visit(node.left)
            for op, right in zip(node.ops, node.comparators):
                self.write(get_op_symbol(op, ' %s '), right)

    # # assignment expressions; new for Python 3.8
    # def visit_NamedExpr(self, node):
    #     with self.delimit(node) as delimiters:
    #         p = delimiters.p
    #         set_precedence(p, node.target)
    #         set_precedence(p + 1, node.value)
    #         # Python is picky about delimiters for assignment
    #         # expressions: it requires at least one pair in any
    #         # statement that uses an assignment expression, even
    #         # when not necessary according to the precedence
    #         # rules. We address this with the kludge of forcing a
    #         # pair of parentheses around every assignment
    #         # expression.
    #         delimiters.discard = False
    #         self.write(node.target, ' := ', node.value)

    def visit_UnaryOp(self, node):
        with self.delimit(node, node.op) as delimiters:
            set_precedence(delimiters.p, node.operand)
            # In Python 2.x, a unary negative of a literal
            # number is merged into the number itself.  This
            # bit of ugliness means it is useful to know
            # what the parent operation was...
            node.operand._p_op = node.op
            sym = get_op_symbol(node.op)
            self.write(sym, ' ' if sym.isalpha() else '', node.operand)

    # def visit_Subscript(self, node):
    #     set_precedence(node, node.slice)
    #     self.write(node.value, '[', node.slice, ']')

    # def visit_Slice(self, node):
    #     set_precedence(node, node.lower, node.upper, node.step)
    #     self.conditional_write(node.lower)
    #     self.write(':')
    #     self.conditional_write(node.upper)
    #     if node.step is not None:
    #         self.write(':')
    #         if not (isinstance(node.step, ast.Name) and
    #                 node.step.id == 'None'):
    #             self.visit(node.step)

    # def visit_Index(self, node):
    #     with self.delimit(node) as delimiters:
    #         set_precedence(delimiters.p, node.value)
    #         self.visit(node.value)

    # def visit_ExtSlice(self, node):
    #     dims = node.dims
    #     set_precedence(node, *dims)
    #     self.comma_list(dims, len(dims) == 1)

    # def visit_Yield(self, node):
    #     with self.delimit(node):
    #         set_precedence(get_op_precedence(node) + 1, node.value)
    #         self.write('yield')
    #         self.conditional_write(' ', node.value)

    # # new for Python 3.3
    # def visit_YieldFrom(self, node):
    #     with self.delimit(node):
    #         self.write('yield from ', node.value)

    # # new for Python 3.5
    # def visit_Await(self, node):
    #     with self.delimit(node):
    #         self.write('await ', node.value)

    # def visit_Lambda(self, node):
    #     with self.delimit(node) as delimiters:
    #         set_precedence(delimiters.p, node.body)
    #         self.write('lambda ')
    #         self.visit_arguments(node.args)
    #         self.write(': ', node.body)

    # def visit_Ellipsis(self, node):
    #     self.write('...')

    # def visit_ListComp(self, node):
    #     with self.delimit('[]'):
    #         self.write(node.elt, *node.generators)

    # def visit_GeneratorExp(self, node):
    #     with self.delimit(node) as delimiters:
    #         if delimiters.pp == Precedence.call_one_arg:
    #             delimiters.discard = True
    #         set_precedence(Precedence.Comma, node.elt)
    #         self.write(node.elt, *node.generators)

    # def visit_SetComp(self, node):
    #     with self.delimit('{}'):
    #         self.write(node.elt, *node.generators)

    # def visit_DictComp(self, node):
    #     with self.delimit('{}'):
    #         self.write(node.key, ': ', node.value, *node.generators)

    # def visit_IfExp(self, node):
    #     with self.delimit(node) as delimiters:
    #         set_precedence(delimiters.p + 1, node.body, node.test)
    #         set_precedence(delimiters.p, node.orelse)
    #         self.write(node.body, ' if ', node.test, ' else ', node.orelse)

    # def visit_Starred(self, node):
    #     self.write('*', node.value)

    # def visit_Repr(self, node):
    #     # XXX: python 2.6 only
    #     with self.delimit('``'):
    #         self.visit(node.value)

    # def visit_Expression(self, node):
    #     self.visit(node.body)

    # # Helper Nodes

    # def visit_arg(self, node):
    #     self.write(node.arg)
    #     self.conditional_write(': ', node.annotation)

    # def visit_alias(self, node):
    #     self.write(node.name)
    #     self.conditional_write(' as ', node.asname)

    # def visit_comprehension(self, node):
    #     set_precedence(node, node.iter, *node.ifs)
    #     set_precedence(Precedence.comprehension_target, node.target)
    #     stmt = ' async for ' if self.get_is_async(node) else ' for '
    #     self.write(stmt, node.target, ' in ', node.iter)
    #     for if_ in node.ifs:
    #         self.write(' if ', if_)


def to_source(node, indent_with=' ' * 4, add_line_information=False,
              pretty_string=pretty_string, pretty_source=pretty_source,
              source_generator_class=None):
    if source_generator_class is None:
        source_generator_class = SourceGenerator
    elif not issubclass(source_generator_class, SourceGenerator):
        raise TypeError('source_generator_class should be a subclass of SourceGenerator')
    elif not callable(source_generator_class):
        raise TypeError('source_generator_class should be a callable')
    generator = source_generator_class(
        indent_with, add_line_information, pretty_string)
    generator.visit(node)
    return pretty_source(generator.get_result())


def main():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    read_file_type = argparse.FileType('r')
    write_file_type = argparse.FileType('w')
    parser.add_argument('file', type=read_file_type, help='Python file to transpile from')
    parser.add_argument('-o', '--output', type=write_file_type, help='Output file path')
    args = parser.parse_args()
    buf = args.file.read()
    astp = ast.parse(buf)
    print(to_source(astp, source_generator_class=CXXSourceGenerator), file=args.output)


if __name__ == "__main__":
    main()
