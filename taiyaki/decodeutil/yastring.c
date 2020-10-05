#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "yastring.h"

#define YASTRING_INITIAL_LENGTH  32

static const yastring const yastring_null = {NULL, 0, 0};

yastring yastring_new(void){
    char * s = calloc(1 + YASTRING_INITIAL_LENGTH, sizeof(char));
    return (yastring){s, 0, (NULL != s) ? YASTRING_INITIAL_LENGTH : 0};
}


void yastring_free(yastring s){
    free(s.str);
}


yastring yastring_copy(const yastring s){
    char * str = calloc(s.cap + 1, sizeof(char));
    if(NULL == str){
        return yastring_null;
    }
    memcpy(str, s.str, s.len * sizeof(char));
    return (yastring){str, s.len, s.cap};
}


yastring yastring_extend(yastring s){
    return yastring_extend2(s, s.cap + s.cap);
}


yastring yastring_extend2(yastring s, size_t new_cap){
    if(s.cap >= new_cap){
        return s;
    }
    char * s2 = realloc(s.str, (new_cap + 1) * sizeof(char));
    memset(s2 + s.len, 0, (new_cap + 1 - s.len) * sizeof(char));
    return (yastring){s2, (NULL != s2) ? s.len : 0, (NULL != s2) ? new_cap : 0};
}


yastring yastring_append(yastring s, char c){
    if(s.len == s.cap){
        s = yastring_extend(s);
    }
    s.str[s.len] = c;
    s.len += 1;
    return s;
}


yastring yastring_join(yastring s, yastring * y, size_t ny, char sep){
    if(NULL == y){return s;}

    size_t total_len = s.len;
    for(size_t i=0 ; i < ny ; i++){
        total_len += y[i].len + 1;
    }
    s = yastring_extend2(s, total_len);

    //  Copy strings into new memory
    for(size_t i=0 ; i < ny ; i++){
        // Add separator
        s.str[s.len] = sep;
        s.len += 1;
        // Copy string
        memcpy(s.str + s.len, y[i].str, y[i].len * sizeof(char));
        s.len += y[i].len;
    }

    return s;
}


yastring yastring_join1(yastring s, yastring y, char sep){
    return yastring_join(s, &y, 1, sep);
}


char yastring_lastchar(const yastring s){
    return s.str[s.len - 1];
}
