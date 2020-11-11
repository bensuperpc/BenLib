#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE vector_max_simd

#include <algorithm>
#include <boost/predef.h>
#include <random>

#if BOOST_COMP_GNUC
extern "C"
{
#    include "quadmath.h"
}
#endif

#include <boost/multiprecision/cpp_int.hpp>
#if BOOST_COMP_GNUC
#    include <boost/multiprecision/float128.hpp>
#endif
#include <boost/test/unit_test.hpp>
#include "crypto/crypto.hpp"

BOOST_AUTO_TEST_CASE(test_crypto_1)
{
    BOOST_REQUIRE_MESSAGE(my::crypto::get_md5hash_from_string("Bonjour") == "ebc58ab2cb4848d04ec23d83f7ddf985",
        my::crypto::get_md5hash_from_string("Bonjour") << " Not equal to: ebc58ab2cb4848d04ec23d83f7ddf985");
    BOOST_REQUIRE_MESSAGE(my::crypto::get_sha1hash_from_string("Bonjour") == "f30ecbf5b1cb85c631fdec0b39678550973cfcbc",
        my::crypto::get_sha1hash_from_string("Bonjour") << " Not equal to: f30ecbf5b1cb85c631fdec0b39678550973cfcbc");
    BOOST_REQUIRE_MESSAGE(my::crypto::get_sha224hash_from_string("Bonjour") == "2ae54077d8b449db24d1cdc284a487292d7055bce5dd03676e66afd4",
        my::crypto::get_sha224hash_from_string("Bonjour") << " Not equal to: 2ae54077d8b449db24d1cdc284a487292d7055bce5dd03676e66afd4");
    BOOST_REQUIRE_MESSAGE(my::crypto::get_sha256hash_from_string("Bonjour") == "9172e8eec99f144f72eca9a568759580edadb2cfd154857f07e657569493bc44",
        my::crypto::get_sha256hash_from_string("Bonjour") << " Not equal to: 9172e8eec99f144f72eca9a568759580edadb2cfd154857f07e657569493bc44");
    BOOST_REQUIRE_MESSAGE(
        my::crypto::get_sha512hash_from_string("Bonjour")
            == "c447dff0d671f62ad580b255b64f7a8f6a30d1b828569cee08b7c861239f8d4856ef38a1166718b045a9713876336c1f623619f6a78fc891d48d0b98c703def3",
        my::crypto::get_sha512hash_from_string("Bonjour")
            << " Not equal to: "
               "c447dff0d671f62ad580b255b64f7a8f6a30d1b828569cee08b7c861239f8d4856ef38a1166718b045a9713876336c1f623619f6a78fc891d48d0b98c703def3");
}

BOOST_AUTO_TEST_CASE(test_crypto_2)
{
    BOOST_REQUIRE_MESSAGE(my::crypto::get_md5hash_from_string("Hello") == "8b1a9953c4611296a827abf8c47804d7",
        my::crypto::get_md5hash_from_string("Hello") << " Not equal to: 8b1a9953c4611296a827abf8c47804d7");
    BOOST_REQUIRE_MESSAGE(my::crypto::get_sha1hash_from_string("Hello") == "f7ff9e8b7bb2e09b70935a5d785e0cc5d9d0abf0",
        my::crypto::get_sha1hash_from_string("Hello") << " Not equal to: f7ff9e8b7bb2e09b70935a5d785e0cc5d9d0abf0");
    BOOST_REQUIRE_MESSAGE(my::crypto::get_sha224hash_from_string("Hello") == "4149da18aa8bfc2b1e382c6c26556d01a92c261b6436dad5e3be3fcc",
        my::crypto::get_sha224hash_from_string("Hello") << " Not equal to: 4149da18aa8bfc2b1e382c6c26556d01a92c261b6436dad5e3be3fcc");
    BOOST_REQUIRE_MESSAGE(my::crypto::get_sha256hash_from_string("Hello") == "185f8db32271fe25f561a6fc938b2e264306ec304eda518007d1764826381969",
        my::crypto::get_sha256hash_from_string("Hello") << " Not equal to: 185f8db32271fe25f561a6fc938b2e264306ec304eda518007d1764826381969");
    BOOST_REQUIRE_MESSAGE(
        my::crypto::get_sha512hash_from_string("Hello")
            == "3615f80c9d293ed7402687f94b22d58e529b8cc7916f8fac7fddf7fbd5af4cf777d3d795a7a00a16bf7e7f3fb9561ee9baae480da9fe7a18769e71886b03f315",
        my::crypto::get_sha512hash_from_string("Hello")
            << " Not equal to: "
               "3615f80c9d293ed7402687f94b22d58e529b8cc7916f8fac7fddf7fbd5af4cf777d3d795a7a00a16bf7e7f3fb9561ee9baae480da9fe7a18769e71886b03f315");
}

BOOST_AUTO_TEST_CASE(test_crypto_3)
{
    BOOST_REQUIRE_MESSAGE(my::crypto::get_md5hash_from_string("") == "d41d8cd98f00b204e9800998ecf8427e",
        my::crypto::get_md5hash_from_string("") << " Not equal to: d41d8cd98f00b204e9800998ecf8427e");
    BOOST_REQUIRE_MESSAGE(my::crypto::get_sha1hash_from_string("") == "da39a3ee5e6b4b0d3255bfef95601890afd80709",
        my::crypto::get_sha1hash_from_string("") << " Not equal to: da39a3ee5e6b4b0d3255bfef95601890afd80709");
    BOOST_REQUIRE_MESSAGE(my::crypto::get_sha224hash_from_string("") == "d14a028c2a3a2bc9476102bb288234c415a2b01f828ea62ac5b3e42f",
        my::crypto::get_sha224hash_from_string("") << " Not equal to: d14a028c2a3a2bc9476102bb288234c415a2b01f828ea62ac5b3e42f");
    BOOST_REQUIRE_MESSAGE(my::crypto::get_sha256hash_from_string("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        my::crypto::get_sha256hash_from_string("") << " Not equal to: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    BOOST_REQUIRE_MESSAGE(
        my::crypto::get_sha512hash_from_string("")
            == "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e",
        my::crypto::get_sha512hash_from_string("")
            << " Not equal to: "
               "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e");
}