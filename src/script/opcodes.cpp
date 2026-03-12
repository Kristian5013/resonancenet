#include "script/opcodes.h"

namespace rnet::script {

std::string_view opcode_name(Opcode op) {
    switch (op) {
        // Constants
        case Opcode::OP_0:              return "OP_0";
        case Opcode::OP_PUSHDATA1:      return "OP_PUSHDATA1";
        case Opcode::OP_PUSHDATA2:      return "OP_PUSHDATA2";
        case Opcode::OP_PUSHDATA4:      return "OP_PUSHDATA4";
        case Opcode::OP_1NEGATE:        return "OP_1NEGATE";
        case Opcode::OP_RESERVED:       return "OP_RESERVED";
        case Opcode::OP_1:              return "OP_1";
        case Opcode::OP_2:              return "OP_2";
        case Opcode::OP_3:              return "OP_3";
        case Opcode::OP_4:              return "OP_4";
        case Opcode::OP_5:              return "OP_5";
        case Opcode::OP_6:              return "OP_6";
        case Opcode::OP_7:              return "OP_7";
        case Opcode::OP_8:              return "OP_8";
        case Opcode::OP_9:              return "OP_9";
        case Opcode::OP_10:             return "OP_10";
        case Opcode::OP_11:             return "OP_11";
        case Opcode::OP_12:             return "OP_12";
        case Opcode::OP_13:             return "OP_13";
        case Opcode::OP_14:             return "OP_14";
        case Opcode::OP_15:             return "OP_15";
        case Opcode::OP_16:             return "OP_16";

        // Flow control
        case Opcode::OP_NOP:            return "OP_NOP";
        case Opcode::OP_VER:            return "OP_VER";
        case Opcode::OP_IF:             return "OP_IF";
        case Opcode::OP_NOTIF:          return "OP_NOTIF";
        case Opcode::OP_VERIF:          return "OP_VERIF";
        case Opcode::OP_VERNOTIF:       return "OP_VERNOTIF";
        case Opcode::OP_ELSE:           return "OP_ELSE";
        case Opcode::OP_ENDIF:          return "OP_ENDIF";
        case Opcode::OP_VERIFY:         return "OP_VERIFY";
        case Opcode::OP_RETURN:         return "OP_RETURN";

        // Stack
        case Opcode::OP_TOALTSTACK:     return "OP_TOALTSTACK";
        case Opcode::OP_FROMALTSTACK:   return "OP_FROMALTSTACK";
        case Opcode::OP_2DROP:          return "OP_2DROP";
        case Opcode::OP_2DUP:           return "OP_2DUP";
        case Opcode::OP_3DUP:           return "OP_3DUP";
        case Opcode::OP_2OVER:          return "OP_2OVER";
        case Opcode::OP_2ROT:           return "OP_2ROT";
        case Opcode::OP_2SWAP:          return "OP_2SWAP";
        case Opcode::OP_IFDUP:          return "OP_IFDUP";
        case Opcode::OP_DEPTH:          return "OP_DEPTH";
        case Opcode::OP_DROP:           return "OP_DROP";
        case Opcode::OP_DUP:            return "OP_DUP";
        case Opcode::OP_NIP:            return "OP_NIP";
        case Opcode::OP_OVER:           return "OP_OVER";
        case Opcode::OP_PICK:           return "OP_PICK";
        case Opcode::OP_ROLL:           return "OP_ROLL";
        case Opcode::OP_ROT:            return "OP_ROT";
        case Opcode::OP_SWAP:           return "OP_SWAP";
        case Opcode::OP_TUCK:           return "OP_TUCK";

        // Splice
        case Opcode::OP_CAT:            return "OP_CAT";
        case Opcode::OP_SUBSTR:         return "OP_SUBSTR";
        case Opcode::OP_LEFT:           return "OP_LEFT";
        case Opcode::OP_RIGHT:          return "OP_RIGHT";
        case Opcode::OP_SIZE:           return "OP_SIZE";

        // Bitwise
        case Opcode::OP_INVERT:         return "OP_INVERT";
        case Opcode::OP_AND:            return "OP_AND";
        case Opcode::OP_OR:             return "OP_OR";
        case Opcode::OP_XOR:            return "OP_XOR";
        case Opcode::OP_EQUAL:          return "OP_EQUAL";
        case Opcode::OP_EQUALVERIFY:    return "OP_EQUALVERIFY";
        case Opcode::OP_RESERVED1:      return "OP_RESERVED1";
        case Opcode::OP_RESERVED2:      return "OP_RESERVED2";

        // Arithmetic
        case Opcode::OP_1ADD:           return "OP_1ADD";
        case Opcode::OP_1SUB:           return "OP_1SUB";
        case Opcode::OP_2MUL:           return "OP_2MUL";
        case Opcode::OP_2DIV:           return "OP_2DIV";
        case Opcode::OP_NEGATE:         return "OP_NEGATE";
        case Opcode::OP_ABS:            return "OP_ABS";
        case Opcode::OP_NOT:            return "OP_NOT";
        case Opcode::OP_0NOTEQUAL:      return "OP_0NOTEQUAL";
        case Opcode::OP_ADD:            return "OP_ADD";
        case Opcode::OP_SUB:            return "OP_SUB";
        case Opcode::OP_MUL:            return "OP_MUL";
        case Opcode::OP_DIV:            return "OP_DIV";
        case Opcode::OP_MOD:            return "OP_MOD";
        case Opcode::OP_LSHIFT:         return "OP_LSHIFT";
        case Opcode::OP_RSHIFT:         return "OP_RSHIFT";
        case Opcode::OP_BOOLAND:        return "OP_BOOLAND";
        case Opcode::OP_BOOLOR:         return "OP_BOOLOR";
        case Opcode::OP_NUMEQUAL:       return "OP_NUMEQUAL";
        case Opcode::OP_NUMEQUALVERIFY: return "OP_NUMEQUALVERIFY";
        case Opcode::OP_NUMNOTEQUAL:    return "OP_NUMNOTEQUAL";
        case Opcode::OP_LESSTHAN:       return "OP_LESSTHAN";
        case Opcode::OP_GREATERTHAN:    return "OP_GREATERTHAN";
        case Opcode::OP_LESSTHANOREQUAL: return "OP_LESSTHANOREQUAL";
        case Opcode::OP_GREATERTHANOREQUAL: return "OP_GREATERTHANOREQUAL";
        case Opcode::OP_MIN:            return "OP_MIN";
        case Opcode::OP_MAX:            return "OP_MAX";
        case Opcode::OP_WITHIN:         return "OP_WITHIN";

        // Crypto
        case Opcode::OP_RIPEMD160:      return "OP_RIPEMD160";
        case Opcode::OP_SHA1:           return "OP_SHA1";
        case Opcode::OP_SHA256:         return "OP_SHA256";
        case Opcode::OP_HASH160:        return "OP_HASH160";
        case Opcode::OP_HASH256:        return "OP_HASH256";
        case Opcode::OP_CODESEPARATOR:  return "OP_CODESEPARATOR";
        case Opcode::OP_CHECKSIG:       return "OP_CHECKSIG";
        case Opcode::OP_CHECKSIGVERIFY: return "OP_CHECKSIGVERIFY";
        case Opcode::OP_CHECKMULTISIG:  return "OP_CHECKMULTISIG";
        case Opcode::OP_CHECKMULTISIGVERIFY: return "OP_CHECKMULTISIGVERIFY";

        // Locktime
        case Opcode::OP_NOP1:           return "OP_NOP1";
        case Opcode::OP_CHECKLOCKTIMEVERIFY: return "OP_CHECKLOCKTIMEVERIFY";
        case Opcode::OP_CHECKSEQUENCEVERIFY: return "OP_CHECKSEQUENCEVERIFY";
        case Opcode::OP_NOP4:           return "OP_NOP4";
        case Opcode::OP_NOP5:           return "OP_NOP5";
        case Opcode::OP_NOP6:           return "OP_NOP6";
        case Opcode::OP_NOP7:           return "OP_NOP7";
        case Opcode::OP_NOP8:           return "OP_NOP8";
        case Opcode::OP_NOP9:           return "OP_NOP9";
        case Opcode::OP_NOP10:          return "OP_NOP10";

        // ResonanceNet extensions
        case Opcode::OP_CHECKHEARTBEAT: return "OP_CHECKHEARTBEAT";
        case Opcode::OP_CHECKRECOVERY:  return "OP_CHECKRECOVERY";
        case Opcode::OP_CHECKGUARDIAN:  return "OP_CHECKGUARDIAN";

        case Opcode::OP_INVALIDOPCODE:  return "OP_INVALIDOPCODE";

        default: return "OP_UNKNOWN";
    }
}

bool is_disabled_opcode(Opcode op) {
    switch (op) {
        case Opcode::OP_CAT:
        case Opcode::OP_SUBSTR:
        case Opcode::OP_LEFT:
        case Opcode::OP_RIGHT:
        case Opcode::OP_INVERT:
        case Opcode::OP_AND:
        case Opcode::OP_OR:
        case Opcode::OP_XOR:
        case Opcode::OP_2MUL:
        case Opcode::OP_2DIV:
        case Opcode::OP_MUL:
        case Opcode::OP_DIV:
        case Opcode::OP_MOD:
        case Opcode::OP_LSHIFT:
        case Opcode::OP_RSHIFT:
            return true;
        default:
            return false;
    }
}

int decode_op_n(Opcode op) {
    auto val = static_cast<uint8_t>(op);
    if (val == 0x00) return 0;
    if (val >= 0x51 && val <= 0x60) {
        return static_cast<int>(val - 0x50);
    }
    return -1;
}

Opcode encode_op_n(int n) {
    if (n == 0) return Opcode::OP_0;
    if (n >= 1 && n <= 16) {
        return static_cast<Opcode>(0x50 + n);
    }
    return Opcode::OP_INVALIDOPCODE;
}

}  // namespace rnet::script
