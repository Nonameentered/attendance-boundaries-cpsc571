#!/usr/bin/env python3

import streamlit as st

import glob
import json
import math
import random
import re

ERROR_ATTRIBUTE_MISSING = "attribute_missing"
ERROR_INVALID_VALUE = "invalid_value"
ERROR_INVALID_CODE = "invalid_code"
ERROR_INVALID_CHECKSUM = "invalid_checksum"
ERROR_BAD_SCHEMA_ID = "bad_schema"


# Map attributes to their max value
CODE_SCHEMAS = {
    0: [
        # district_id
        ("district_id", 6000000),
        # what's the role of the receipient:
        ("stakeholder_type_id", 3),
        # within the role, the ID of the person
        ("stakeholder_index", 31),
        # mailer, email, online ad?
        ("medium_type_id", 3),
        # which messaging campaign
        ("campaign_id", 7),
        # message format
        ("creative_treatment_id", 3),
        # type-in, click on link, scan QR code?
        ("access_type_id", 3),
        # sum of all vals mod 10
        ("checksum", 10),
    ]
}


LEVELS = {
    "stakeholder_type": [
        "unknown",
        "school_board_member",
        "superindentendent",
        "parent",
    ],
    "medium_type": ["unknown", "postal", "email"],
    "access_type": ["no_code", "typein", "click", "qr_scan"],
}


@st.cache_data
def load_campaign_info():
    info = {}  # referral_code -> whole record
    filenames = glob.glob("../outreach/campaign*.jsonl")
    for fn in filenames:
        with open(fn) as fs_in:
            for line in fs_in:
                x = json.loads(line)
                info[x["code"]["referral_code"]] = x
    return info


def referral_code_to_attributes(code):
    schema = 0  # assume 0 for now
    if not re.match("^[0-9]+$", code) or len(code) <= 1:
        return {"error": ERROR_INVALID_CODE}
    checksum = int(code[-1:])
    if checksum != sum(int(x) for x in code[:-1]) % 10:
        return {"error": ERROR_INVALID_CHECKSUM}

    code = bin(int(code[:-1]))[2:][1:]  # remove checksum, skip first bit

    res = {}
    for att, val_max in CODE_SCHEMAS[schema]:
        if att == "checksum":
            continue
        num_bits = math.ceil(math.log2(val_max))
        if len(code) < num_bits:
            return {"error": ERROR_INVALID_CODE}
        val = int(code[0:num_bits], 2)
        res[att] = val
        code = code[num_bits:]
    return res


def attributes_to_referral_code(attributes, schema=0):
    referral_code = ""
    for att, val_max in CODE_SCHEMAS[schema]:
        if att == "schema_id":
            continue
        elif att == "checksum":
            referral_code = str(int("1" + referral_code, 2))
            checksum = sum(int(x) for x in referral_code if x.isdigit())
            referral_code += str(checksum % 10).zfill(1)
            continue
        try:
            att_index = int(attributes[att])
        except ValueError:
            return {"error": ERROR_INVALID_VALUE + " " + att + " " + attributes[att]}
        except KeyError:
            att = att.replace("_id", "")
            if att in LEVELS and att in attributes:
                if attributes[att] in LEVELS[att]:
                    att_index = LEVELS[att].index(attributes[att])
                else:
                    return {
                        "error": ERROR_INVALID_VALUE + " " + att + " " + attributes[att]
                    }
            else:
                # attribute not included
                return {"error": ERROR_ATTRIBUTE_MISSING + " " + att}

        if att_index > val_max or att_index < 0:
            return {"error": ERROR_INVALID_VALUE + " " + att + " " + str(att_index)}
        num_bits = math.ceil(math.log2(val_max))
        referral_code += str(bin(att_index)[2:].zfill(num_bits))

    if schema == 0:
        return {"referral_code": referral_code}
    else:
        return {"error": ERROR_BAD_SCHEMA_ID}


def is_referral_code_valid(code):
    return "error" not in referral_code_to_attributes(code)


if __name__ == "__main__":
    # Test a bunch of examples
    for i in range(1000):
        test_atts = {
            "district_id": random.randint(0, 6000000),
            "stakeholder_type_id": random.randint(0, 3),
            "stakeholder_index": random.randint(0, 31),
            "medium_type_id": random.randint(0, 3),
            "creative_treatment_id": random.randint(0, 3),
            "access_type_id": random.randint(0, 3),
            "campaign_id": 0,
        }
        x = attributes_to_referral_code(test_atts)
        code = x["referral_code"]
        assert is_referral_code_valid(code)
        assert test_atts == referral_code_to_attributes(code)
