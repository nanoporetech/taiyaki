#pragma once

#ifndef YASTRING_H
#define YASTRING_H


#include <stdlib.h>

typedef struct _yastring {
    char * str;       // Memory to store string
    size_t len, cap;  // length of string and total capacity
} yastring;


yastring yastring_new(void);
void yastring_free(yastring s);

yastring yastring_copy(const yastring s);
yastring yastring_extend(yastring s);
yastring yastring_extend2(yastring s, size_t new_cap);

yastring yastring_append(yastring s, char c);
yastring yastring_join(yastring s, yastring * y, size_t ny, char sep);
yastring yastring_join1(yastring s, yastring y, char sep);

char yastring_lastchar(const yastring s);

#endif /* YASTRING_H */
