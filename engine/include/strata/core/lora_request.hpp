// include/strata/core/lora_request.hpp - whether `lora=` is allowed on one request line, as pure functions.
//
// WHY A HEADER OF ITS OWN.  The refusals have to be pinned by a test that runs without a GPU, without a model and
// without an engine (`src/core/lora_pack_test.cpp`), so the decisions cannot live inline in the request parser of
// `src/program/generate.cpp`, where exercising them means starting an engine.  Nothing here includes CUDA, the
// geometry or the adapter: plain arguments in, one string out.
//
// THE RULE, and why it is a refusal rather than a no-op.  `lora=1` is a DECISION request's key: the adapted model
// answers the LOGITS line and the chat turn beside it runs the base model, which is what makes the phase 2 exit
// criterion ("the chat turn is byte-identical with and without the adapter") true by construction rather than by
// measurement.  A `lora=1` on a chat line asks for the one thing that criterion forbids, so it is refused with the
// rule named - a silent no-op would leave a caller believing its chat turn carried the adapter.
//
// WHY THE VALUE IS PARSED HERE AND NOT WITH `atoi`.  This key selects WHICH MODEL answers, so a value the engine did
// not understand must be an error, not an implicit "base": `lora=yes`, `lora=true` and a bare `lora=` all read as 0
// through `std::atoi`, and a caller that typed one of them would be answered by the base weights with nothing said.
// Only `0` and `1` are accepted, and a line that gives the key twice is refused (one request cannot carry two
// values, and "the last one wins" would be another silent choice).
//
// WHY `loaded` IS AN ARGUMENT.  C3 starts the engine WITHOUT an adapter when `--lora` names an adapter the loader
// refused (plan 3.5), and an engine started without `--lora` has none either.  A `lora=1` on such an engine must be
// refused too: honoring it by falling back to the base weights answers the decision with the wrong distribution and
// reports no error.  The caller passes `kernels::lora().loaded()`, so this header still knows nothing about the
// device.
#pragma once

#include <string>

namespace strata::core {

/// The key's own syntax: "" when `lora=` may be read, otherwise the sentence to send back.  `asked` is whether the
/// line named the key at all (an absent key is not this function's business); `repeated` is whether it appeared more
/// than once; `value` is the text after `lora=`, which may be empty.
inline std::string lora_key_problem(bool asked, bool repeated, const std::string& value) {
    if (!asked) return std::string();
    if (repeated)
        return "lora= is given twice on one line: say it once, 0 or 1 (this engine holds one adapter)";
    if (value == "0" || value == "1") return std::string();
    return "lora= takes 0 or 1, got '" + value + "' (0 = the base model, 1 = the loaded adapter)";
}

/// The request the key describes: "" when it may be honored, otherwise the sentence to send back.  `asked` is whether
/// the line named the key with a nonzero value; `req_logits_k` is the LOGITS parser's k, and 0 on every other line
/// (GEN, GENI, and BGEN, which is rewritten into `GEN 1` before the keys are read); `loaded` is whether an adapter is
/// resident.
///
/// The LINE is checked before the ADAPTER, deliberately: a chat line with no adapter has the wrong key on it, and
/// telling that caller to load an adapter would send it looking for the wrong fix.
inline std::string lora_request_problem(bool asked, int req_logits_k, bool loaded) {
    if (!asked) return std::string();
    if (req_logits_k <= 0)
        return "lora=1 needs a LOGITS line: the adapted model answers decisions, while a chat turn runs the base "
               "model (use LOGITS <k> <max_new> <ids> [lora=1])";
    if (!loaded)
        return "lora=1 needs a loaded adapter: the engine was started without --lora, or its adapter was refused at "
               "start-up - see INFO lora=";
    return std::string();
}

}  // namespace strata::core
