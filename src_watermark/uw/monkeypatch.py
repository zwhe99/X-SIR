#!/usr/bin/env python
# -*- coding: utf-8 -*-

import types
from transformers import GenerationMixin, GenerationConfig
from transformers import LogitsProcessorList


def patch_model(model: GenerationMixin):
    original_generate = model.generate
    original__get_logits_warper = model._get_logits_warper

    context = {}

    def generate(self, *args, logits_warper=None, **kargs):
        logits_warper = (
            logits_warper if logits_warper is not None else LogitsProcessorList()
        )
        context["logits_warper"] = logits_warper
        return original_generate(*args, **kargs)

    def _get_logits_warper(self, *args, **kargs):
        warpers = original__get_logits_warper(*args, **kargs)
        if "logits_warper" in context:
            warpers = self._merge_criteria_processor_list(
                warpers, context["logits_warper"]
            )
        return warpers

    def _clear_patch_context(self):
        context.clear()

    model.generate = types.MethodType(generate, model)
    model._get_logits_warper = types.MethodType(_get_logits_warper, model)
    model._clear_patch_context = types.MethodType(_clear_patch_context, model)
