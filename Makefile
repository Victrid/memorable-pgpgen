all: vanity
clean:
	$(RM) vanity *.o core a.out

OPENSSL_CFLAGS := `pkg-config --cflags openssl`
OPENSSL_LIBS := `pkg-config --libs openssl`

vanity: vanity.c
	$(CC) \
		-o $@ \
		-O3 \
		-W \
		-Wall \
		$< \
		$(OPENSSL_CFLAGS) \
		$(OPENSSL_LIBS) \


