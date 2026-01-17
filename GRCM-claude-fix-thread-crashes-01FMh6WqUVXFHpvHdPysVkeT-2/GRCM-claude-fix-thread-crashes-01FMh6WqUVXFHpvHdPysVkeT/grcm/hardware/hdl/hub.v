// ============================================================
// EchoZero Hub Node
// Implements global coupling constraint: -λΣψ
// ============================================================

module echozero_hub #(
    parameter DATA_WIDTH = 32,
    parameter LAMBDA = 16'h0051  // 0.02 in fixed-point
)(
    input wire clk,
    input wire rst_n,

    // Bus interface
    input wire [DATA_WIDTH-1:0] sum_psi,
    output reg [DATA_WIDTH-1:0] hub_output,

    // Control
    input wire enable,
    output reg valid
);

    // Hub constraint: -λΣψ
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            hub_output <= 32'h0;
            valid <= 1'b0;
        end else if (enable) begin
            // Multiply sum by -λ
            hub_output <= multiply_lambda(sum_psi);
            valid <= 1'b1;
        end else begin
            valid <= 1'b0;
        end
    end

    // Fixed-point multiplication: -λ * sum
    function [DATA_WIDTH-1:0] multiply_lambda;
        input [DATA_WIDTH-1:0] value;
        reg signed [DATA_WIDTH-1:0] result;
        begin
            // Simplified: result = -lambda * value
            result = -($signed(LAMBDA) * $signed(value)) >>> 10;  // Fixed-point shift
            multiply_lambda = result;
        end
    endfunction

endmodule
