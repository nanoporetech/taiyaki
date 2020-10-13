#include <stdlib.h>
#include <string.h>

#include "yastring.h"

#define YASTRING_INITIAL_LENGTH  32

//  An empty string
static const yastring const yastring_null = {NULL, 0, 0};

/**  Create a new string
 *
 *   @returns new yastring
 **/
yastring yastring_new(void){
    //  Allocate one byte more than capacity so string is always NULL
    //  terminated
    char * s = calloc(1 + YASTRING_INITIAL_LENGTH, sizeof(char));
    return (yastring){s, 0, (NULL != s) ? YASTRING_INITIAL_LENGTH : 0};
}


/*   Frees a string
 *
 *   Contents of string may be NULL, e.g. yastring_null, in which
 *     case nothing happens.  After the string is freed, the pointer
 *     stored inside the `yastring` type is nolonger valid.
 *
 *   @param s       String to free
 *
 *   @returns void
 **/
void yastring_free(yastring s){
    free(s.str);
}


/**  Copies a string
 *
 *   @param s       String to copy
 *
 *   @returns new string containing copy of input
 **/
yastring yastring_copy(const yastring s){
    char * str = calloc(s.cap + 1, sizeof(char));
    if(NULL == str){
        return yastring_null;
    }
    memcpy(str, s.str, s.len * sizeof(char));
    return (yastring){str, s.len, s.cap};
}



/**  Doubles buffer for string
 *
 *   If memory can't be increased at the current location, new memory
 *     is allocated and the string copied (see `alloc` for details). In
 *     these circumstances, the pointer contained in the input is no
 *     longer valid.
 *
 *   @example
 *      s = yastring_extend(s);
 *
 *   @param s       String to copy
 *
 *   @returns new string containing copy of input
 **/
yastring yastring_extend(yastring s){
    return yastring_extend2(s, s.cap + s.cap);
}


/**  Increases buffer for string
 *
 *   If new capacity is less than old, the input string is returned.
 *
 *   If memory can't be increased at the current location, new memory
 *     is allocated and the string copied (see `alloc` for details). In
 *     these circumstances, the pointer contained in the input is no
 *     longer valid.
 *
 *   @example
 *      s = yastring_extend2(s, s.cap + 20);
 *
 *   @param s       String to copy
 *   @param new_cap New capacity of string
 *
 *   @returns new string containing copy of input
 **/
yastring yastring_extend2(yastring s, size_t new_cap){
    if(s.cap >= new_cap){
        return s;
    }
    //  Allocate 1 byte more than capacity, so string is NULL terminated
    char * s2 = realloc(s.str, (new_cap + 1) * sizeof(char));
    //  All new memory is set to  zero (NULL)
    memset(s2 + s.len, 0, (new_cap + 1 - s.len) * sizeof(char));
    return (yastring){s2, (NULL != s2) ? s.len : 0, (NULL != s2) ? new_cap : 0};
}


/**  Append character to end of string
 *
 *   If the string has insufficient capacity to append character, it
 *      is extended to double capacity and the new string returned.
 *
 *   @example
 *      s = yastring_append(s, 'a');
 *
 *   @param s       String to append to
 *   @param c       Character to append
 *
 *   @returns string with character appended.  Memory may be reallocated.
 **/
yastring yastring_append(yastring s, char c){
    if(s.len == s.cap){
        s = yastring_extend(s);
    }
    s.str[s.len] = c;
    s.len += 1;
    return s;
}


/**  Join multiple strings together interspersed with separator
 *
 *   Uses spare capacity in `s` if possible, otherwises extends.
 *   Pointers in input string may be invalidated.
 *
 *   @param s       String to join onto
 *   @param y       Array[ny] of pointers to strings to append
 *   @param ny      Number of strings to append
 *   @param sep     Character to separate strings by
 *
 *   @returns string contained input, memory may have been reallocated if
 *      input string `s` did not have sufficent capacity.
 **/
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


/**  Join two strings together interspersed with separator
 *
 *   Uses spare capacity in `s` if possible, otherwises extends.
 *   Pointers in input string may be invalidated.
 *
 *   @param s       String to join onto
 *   @param y       Array[ny] of pointers to strings to append
 *   @param sep     Character to separate strings by
 *
 *   @returns string contained input, memory may have been reallocated if
 *      input string `s` did not have sufficent capacity.
 **/
yastring yastring_join1(yastring s, yastring y, char sep){
    return yastring_join(s, &y, 1, sep);
}



/**  Last character of string
 *
 *   @param s       String to get last character from
 *
 *   @returns character
 **/
char yastring_lastchar(const yastring s){
    return s.str[s.len - 1];
}
