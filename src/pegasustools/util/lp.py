# Overwrite dimod lp.dump to be readable by gurobi when offset is non-zero

import collections.abc
import functools
import io
import os
import shutil
import string
import tempfile
import typing

import dimod  # for typing

from dimod.cylp import cyread_lp_file
from dimod.sym import Sense
from dimod.vartypes import Vartype

from dimod.lp import _validate_label, _WidthLimitedFile, _sign, _abs, _sense


def dump(cqm: dimod.ConstrainedQuadraticModel, file_like: typing.TextIO):
    """Serialize a constrained quadratic model as an LP file.

    LP files are a common format for encoding optimization models. See
    documentation from Gurobi_ and CPLEX_.

    Args:
        cqm: A constrained quadratic model.
        file_like: A ``.write()`` supporting file-like_ object.

    .. _file-like: https://docs.python.org/3/glossary.html#term-file-object

    .. _Gurobi: https://www.gurobi.com/documentation/9.5/refman/lp_format.html

    .. _CPLEX: https://www.ibm.com/docs/en/icos/12.8.0.0?topic=cplex-lp-file-format-algebraic-representation

    """
    # check that there are no soft constraints, LP format does not support them
    if len(cqm._soft) > 0:
        raise ValueError(f"LP file does not support soft constraints, {len(cqm._soft)} were given")

    # check that constraint labels are serializable
    for c in cqm.constraints:
        _validate_label(c)

    vartypes = {var: cqm.vartype(var) for var in cqm.variables}

    # check that variable labels and types are serializable
    for v, vartype in vartypes.items():
        _validate_label(v)

        if vartype == Vartype.SPIN:
            raise ValueError(
                'SPIN variables not supported in LP files, convert them to BINARY beforehand.')

    f = _WidthLimitedFile(file_like)

    # write the objective
    for var, bias in cqm.objective.iter_linear():
        if bias:
            if not f.tell():
                # LP files allow to omit this part if the objective is empty, so
                # we add these lines if we are sure that there are nonzero terms.
                f.write("Minimize\n")
                f.write(' obj: ')
            f.write(f"{_sign(bias)} {_abs(bias)} {var} ")

    if not cqm.objective.is_linear():
        if not f.tell():
            # if the objective is only quadratic then the header is written now
            f.write("Minimize\n")
            f.write(' obj: ')

        f.write('+ [ ')
        for u, v, bias in cqm.objective.iter_quadratic():
            # multiply bias by two because all quadratic terms are eventually
            # divided by two outside the squared parenthesis
            f.write(f"{_sign(bias)} {_abs(2 * bias)} {u} * {v} ")
        f.write(']/2 ')

    if cqm.objective.offset:
        if not f.tell():
            # if the objective has only an offset then the header is written now
            f.write("Minimize\n")
            f.write(' obj: ')
        f.write(f"+ offset")

    # write the constraints
    f.write("\n\n")
    f.write("Subject To \n")

    for label, constraint in cqm.constraints.items():
        f.write(f' {label}: ')

        for var, bias in constraint.lhs.iter_linear():
            if bias:
                f.write(f"{_sign(bias)} {_abs(bias)} {var} ")

        if constraint.lhs.quadratic:
            f.write('+ [ ')
            for u, v, bias in constraint.lhs.iter_quadratic():
                f.write(f"{_sign(bias)} {_abs(bias)} {u} * {v} ")
            f.write('] ')

        rhs = constraint.rhs - constraint.lhs.offset
        f.write(f" {_sense(constraint.sense)} {rhs}\n")

    # write variable bounds
    f.write('\n')
    f.write('Bounds\n')
    if cqm.objective.offset:
        offset = cqm.objective.offset
        f.write(f' {offset} <= offset <= {offset}\n')
    for v, vartype in vartypes.items():
        if vartype in (Vartype.INTEGER, Vartype.REAL):
            f.write(f' {cqm.lower_bound(v)} <= {v} <= {cqm.upper_bound(v)}\n')
        elif vartype is not Vartype.BINARY:
            raise RuntimeError(f"unexpected vartype {vartype}")

    # write variable names
    for section, vartype_ in (('Binary', Vartype.BINARY), ('General', Vartype.INTEGER)):
        f.write('\n')
        f.write(f'{section}\n')

        for v, vartype in vartypes.items():
            if vartype is vartype_:
                f.write(f' {v}')

    # conclude
    f.write('\n')
    f.write('End')
    return f