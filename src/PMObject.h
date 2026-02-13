#ifndef PMOBJECT_H_INCLUDED
#define PMOBJECT_H_INCLUDED

#include <Python.h>

//---------------------------------------------------------------
// Wrapper of PyObject with reference counting handling.
// Given
//     PMObject pmobject
//
// To access PyObject:
//     pmobject()
//
// For a new reference:
//     PMObject pmobject(py_obj, true)
//
// For a borrowed reference:
//     PMObject pmobject(py_obj, false)
//
// Before the reference is stolen:
//     pmobject.to_be_stolen()
//---------------------------------------------------------------

template<class T>
class PMObject {
public:
    PMObject() {}
    PMObject(T obj, bool protect = false) { assign(obj, protect); }
    PMObject(const PMObject &other) { *this = other; }

    ~PMObject() { clear(); }

    void clear();

    PMObject &operator=(const PMObject &other);
    const T  &operator*() const { return m_obj; }
    T        &operator*() { return m_obj; }
    bool      operator!() { return !m_obj; }

    operator       T() { return m_obj; }
    operator const T() const { return m_obj; }

    void to_be_stolen();   // call this function before the reference is stolen

    PMObject &assign(T obj, bool protect = false);

private:
    T    m_obj{NULL};
    bool m_protect{false};

    PMObject &operator=(const PyObject *) = delete;
};

typedef PMObject<PyObject *> PMPY;
typedef PMObject<PyArrayObject *> PMPYARR;


//------------------------------------ IMPLEMENTATION ----------------------------------------

template <class T>
void PMObject<T>::clear()
{
    if (m_obj && m_protect)
        Py_DECREF(m_obj);
    m_obj = NULL;;
    m_protect = false;
}

template <class T>
PMObject<T> &PMObject<T>::operator=(const PMObject<T> &other)
{
    clear();
    m_obj = other.m_obj;
    m_protect = other.m_protect;
    if (m_obj && m_protect)
        Py_INCREF(m_obj);
    return *this;
}

template <class T>
void PMObject<T>::to_be_stolen()
{
    if (m_obj)
        Py_INCREF(m_obj);
}

template <class T>
PMObject<T> &PMObject<T>::assign(T obj, bool protect)
{
    clear();
    m_obj = obj;
    m_protect = protect;
    return *this;
}

#endif
